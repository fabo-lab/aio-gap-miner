#!/usr/bin/env python3
"""Build the narrative notebook programmatically with nbformat.

Run, then execute with:
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_gap_miner_baseline.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_gap_miner_baseline.ipynb"

cells = []
def md(t): cells.append(new_markdown_cell(t.strip("\n")))
def code(t): cells.append(new_code_cell(t.strip("\n")))

# --------------------------------------------------------------------------- #
md(r"""
# AIO Gap-Miner — predicting AI Overview citations

**Problem.** Google's AI Overviews (and ChatGPT / Perplexity / Claude) answer
informational queries at the top of the page and cite a *handful* of sources.
If your URL isn't in that citation set, your ranking position barely matters —
the click never happens. This project predicts, for a given query, **which
candidate URLs get cited**, and — via SHAP — **explains why**.

**Unit of observation.** One row per **(query, URL)** pair. Label `cited = 1`
if the URL was cited in the AI Overview for that query, else `0`.

**Pipeline (this notebook):**
`SQLite/SQLAlchemy ETL → EDA (seaborn) → inferential statistics → feature
engineering → GroupKFold CV (LightGBM + Logistic Regression) → evaluation →
TreeSHAP → Tableau hand-off.`

| Decision | Choice | Reason |
|---|---|---|
| Task | Binary classification over (query, URL) pairs | Citation is a per-candidate yes/no |
| Models | **Logistic Regression** + **LightGBM** | Transparent linear baseline vs non-linear tree model |
| Validation | **GroupKFold** grouped by `query_id` | Labels are query-relative → no query may leak across the split |
| Metric | **PR-AUC** (average precision) | Positives are rare (~17%) and query-relative; ROC-AUC/accuracy flatter the model |
| Explainability | **TreeSHAP** | Exact per-feature attribution → auditable "why" |

> The committed sample data is **synthetic** (see `data/sample/`) so this
> notebook runs for anyone with zero data access. The numbers demonstrate the
> *method*, not real-world findings. Drop a real labelled CSV with the same
> schema into `data/raw/` to reproduce on live data.
""")

md("## 1 — Setup")
code(r"""
%matplotlib inline
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from aio_gap_miner import config
from aio_gap_miner.data import load_dataset
from aio_gap_miner.database import (
    build_database, read_sql,
    QUERY_CITATION_RATE_BY_CONTENT_TYPE, QUERY_CITATION_RATE_BY_RANK_BUCKET,
)
from aio_gap_miner.features import build_xy
from aio_gap_miner.model import (
    run_group_kfold_cv, run_logreg_group_kfold_cv, train_final_model,
)
from aio_gap_miner import stats as st
from aio_gap_miner import evaluate as ev
from aio_gap_miner import explain as ex

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 140)
np.random.seed(config.RANDOM_SEED)
print("aio_gap_miner ready — features:", len(config.FEATURES))
""")

md(r"""
## 2 — ETL: load into SQLite (SQLAlchemy), query with SQL

The flat CSV is loaded into a local SQLite database via SQLAlchemy, then the
working set is read back with SQL. In production the SQLite URL swaps for
Postgres/BigQuery and nothing else changes.
""")
code(r"""
raw = load_dataset()                       # synthetic sample
engine = build_database(raw)               # ETL -> SQLite
df = read_sql("SELECT * FROM candidates", engine)
print(f"{len(df):,} rows | {df[config.GROUP_COL].nunique():,} queries "
      f"| positive rate {df[config.TARGET].mean():.1%}")
df.head()
""")

md("### 2.1 Analytical SQL — citation rate by rank bucket and content type")
code(r"""
display(read_sql(QUERY_CITATION_RATE_BY_RANK_BUCKET, engine))
read_sql(QUERY_CITATION_RATE_BY_CONTENT_TYPE, engine)
""")

md("## 3 — Exploratory data analysis")
md("### 3.1 Class balance and candidates per query\n"
   "Citation is rare and query-relative — the regime where PR-AUC is the honest "
   "metric.")
code(r"""
per_query = df.groupby(config.GROUP_COL).agg(
    candidates=("url", "size"), cited=(config.TARGET, "sum"))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(x=df[config.TARGET], palette=["#7a8b8b", "#d5602e"], ax=axes[0])
axes[0].set_title(f"Class balance — {df[config.TARGET].mean():.1%} cited")
axes[0].set_xticklabels(["not cited", "cited"])
sns.histplot(per_query["candidates"], bins=20, color="#7a8b8b", ax=axes[1])
axes[1].set_title("Candidate URLs per query")
sns.histplot(per_query["cited"], bins=range(0, 10), color="#d5602e", ax=axes[2])
axes[2].set_title("Citations per query")
plt.tight_layout(); plt.show()
print(per_query.describe().round(2))
""")

md("### 3.2 Feature correlation (seaborn heatmap)")
code(r"""
st.plot_correlation_heatmap(df); plt.show()
""")

md("### 3.3 How signals differ between cited and non-cited URLs")
code(r"""
st.plot_signal_distributions(df); plt.show()
""")

md(r"""
## 4 — Inferential statistics: do the groups really differ?

The A/B-testing mindset on observational data: treat *cited* vs *not cited* as
two groups and test where they diverge. Features are skewed and non-normal, so
we use the **Mann-Whitney U** test (not a t-test) and report a rank-biserial
**effect size** so significance on ~7k rows isn't mistaken for a large effect.
""")
code(r"""
tests = st.hypothesis_tests(df)
tests.round({"median_cited": 3, "median_not_cited": 3,
             "u_statistic": 0, "p_value": 6, "effect_size_r": 3})
""")
md("Descriptive summary by class (mean / median / std):")
code(r"""
st.descriptive_by_class(df)
""")

md("## 5 — Feature engineering\n"
   "`build_xy` adds `rank_reciprocal` (1/rank, the non-linear visibility decay) "
   "and `structure_score` (a single 'how extractable is this page' signal) and "
   "casts categoricals to pandas `category` dtype for LightGBM.")
code(r"""
X, y, groups = build_xy(df)
print("X shape:", X.shape, "| any NaNs:", bool(X.isna().any().any()))
print("categoricals:", [c for c in X.columns if str(X[c].dtype) == 'category'])
X.head(3)
""")

md(r"""
## 6 — Leakage-safe cross-validation: two models

`GroupKFold(groups=query_id)` keeps every query wholly inside one fold. We train
a transparent **logistic regression** (scaled numerics + one-hot categoricals)
and a **LightGBM** tree model on the *same* folds, and collect out-of-fold
predictions for both.
""")
code(r"""
cv = run_group_kfold_cv(X, y, groups, verbose=True)
oof_lr = run_logreg_group_kfold_cv(X, y, groups)
print(f"\nLightGBM per-fold PR-AUC: {cv.mean_ap:.4f} +/- {cv.std_ap:.4f}")
""")

md(r"""
## 7 — Evaluation vs baselines

Scored per-fold on the shared GroupKFold splits (mean ± std), so the comparison
is apples-to-apples. The bar to beat is the **rank-only heuristic** (predict
citation from organic position alone).
""")
code(r"""
comparison = ev.compare_models(df, {
    "Gap-Miner (LightGBM)": cv.oof_pred,
    "Logistic Regression": oof_lr,
}, groups)
comparison
""")
md("**Read-out.** Both learned models beat the rank-only heuristic by ~10 PR-AUC "
   "points and lift per-query precision@k. On this synthetic data the label is "
   "close to linear in the engineered features, so logistic regression is very "
   "competitive; gradient boosting's edge typically grows with the non-linear "
   "interactions present in real citation data. LightGBM is carried forward for "
   "SHAP because tree attributions are exact.")

md("### 7.1 Precision-Recall curve")
code(r"""
ev.plot_pr_curves(df, cv.oof_pred); plt.show()
""")

md("### 7.2 Confusion matrix at the F1-optimal threshold")
code(r"""
summary = ev.evaluation_summary(df, cv.oof_pred)
ev.plot_confusion(df, cv.oof_pred, threshold=summary["best_f1_threshold"]); plt.show()
print(f"Operating threshold: {summary['best_f1_threshold']:.3f} | F1 = {summary['best_f1']:.3f}")
""")

md("## 8 — Explainability (TreeSHAP)\n"
   "Why does a URL get cited? Train a final LightGBM on all rows (at the mean "
   "best CV iteration) and attribute predictions with TreeSHAP.")
code(r"""
final_model = train_final_model(X, y, n_estimators=int(np.mean(cv.best_iterations)))
explainer, shap_values = ex.compute_shap_values(final_model, X)
importance = ex.mean_abs_importance(shap_values, list(X.columns))
importance.head(10)
""")
md("### 8.1 Global importance and direction (beeswarm)")
code(r"""
ex.plot_beeswarm(shap_values, X); plt.show()
""")
md("### 8.2 Response curve for the top driver")
code(r"""
top_feature = importance.iloc[0]["feature"]
ex.plot_dependence(shap_values, X, top_feature); plt.show()
print("Top driver:", top_feature)
""")

md(r"""
## 9 — Read-out, Tableau hand-off & next steps

**What the method shows (synthetic data).** Both models beat a strong rank-only
heuristic on PR-AUC and per-query precision@k, and SHAP attributes the edge to
**semantic passage match, content structure, and domain citation history** —
signals raw ranking position can't see. That is the GEO thesis made measurable:
*structured, on-topic pages get cited beyond what their SERP position predicts.*

**Tableau hand-off.** `python scripts/export_tableau.py` writes a flat, scored
extract (`tableau/aio_gap_miner_tableau.csv`) with both models' probabilities, a
**citation-gap flag** (predicted-cited but not yet cited = the opportunity), and
the strongest SHAP driver per row. The interactive dashboard is built on that
(see `tableau/README.md`).

**From here:**
1. **Real labels** — swap the synthetic sample for labelled AI Overview citation
   data (same schema). No pipeline changes.
2. **Richer features** — real embeddings for query↔passage similarity, NER entity
   coverage, SERP-feature flags, Core Web Vitals.
3. **Calibration** — isotonic/Platt on top of the ranker for an absolute
   "citation likelihood" score.
4. **Gap reports** — per-page SHAP names the specific levers (add FAQ schema,
   tighten the answer passage, raise topical coverage) to close a citation gap —
   the analytical layer a rules-based tool can't offer.
""")

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Wrote {OUT} with {len(cells)} cells")
