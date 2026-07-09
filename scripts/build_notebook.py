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


def md(text: str) -> None:
    cells.append(new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(new_code_cell(text.strip("\n")))


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

**Why this is the right framing.**

| Decision | Choice | Reason |
|---|---|---|
| Task | Binary classification over (query, URL) pairs | Citation is a per-candidate yes/no |
| Model | LightGBM (gradient-boosted trees) | Tabular, mixed feature types, non-linear, fast |
| Validation | **GroupKFold** grouped by `query_id` | Labels are query-relative → no query may leak across the split |
| Metric | **PR-AUC** (average precision) | Positives are rare (~17%) and query-relative; ROC-AUC/accuracy flatter the model |
| Explainability | **TreeSHAP** | Exact per-feature attribution → auditable "why", the differentiator vs a black box |

> The committed sample data is **synthetic** (see `data/sample/`) so this
> notebook runs for anyone with zero data access. The numbers below therefore
> demonstrate the *method*, not real-world findings. Drop a real labelled CSV
> with the same schema into `data/raw/` to reproduce on live data.
""")

md("## 1 — Setup")
code(r"""
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aio_gap_miner import config
from aio_gap_miner.data import load_dataset
from aio_gap_miner.features import build_xy
from aio_gap_miner.model import run_group_kfold_cv, train_final_model
from aio_gap_miner import evaluate as ev
from aio_gap_miner import explain as ex

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)
RANDOM_SEED = config.RANDOM_SEED
np.random.seed(RANDOM_SEED)
print("aio_gap_miner ready — features:", len(config.FEATURES))
""")

md("## 2 — Load the data\nOne row per (query, URL) candidate.")
code(r"""
df = load_dataset()  # defaults to the committed synthetic sample
print(f"{len(df):,} rows | {df[config.GROUP_COL].nunique():,} queries "
      f"| positive rate {df[config.TARGET].mean():.1%}")
df.head()
""")

md("## 3 — Exploratory analysis")
md("### 3.1 Class balance and candidates per query\n"
   "Citation is rare and query-relative — this is exactly the regime where "
   "PR-AUC is the honest metric.")
code(r"""
per_query = df.groupby(config.GROUP_COL).agg(
    candidates=("url", "size"),
    cited=(config.TARGET, "sum"),
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df[config.TARGET].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], color=["#c7c1b8", "#d5602e"])
axes[0].set_title(f"Class balance — {df[config.TARGET].mean():.1%} cited")
axes[0].set_xticklabels(["not cited (0)", "cited (1)"], rotation=0)

per_query["candidates"].plot(kind="hist", bins=20, ax=axes[1], color="#7a8b8b")
axes[1].set_title("Candidate URLs per query")
axes[1].set_xlabel("candidates")

per_query["cited"].plot(kind="hist", bins=range(0, 10), ax=axes[2], color="#d5602e")
axes[2].set_title("Citations per query")
axes[2].set_xlabel("cited URLs")
plt.tight_layout(); plt.show()

print(per_query.describe().round(2))
""")

md("### 3.2 Which raw signals separate cited from non-cited?\n"
   "A first, model-free look: mean feature value by class. Large gaps hint at "
   "predictive features (semantic match, structure, rank).")
code(r"""
signal_cols = ["organic_rank", "query_url_similarity", "passage_match_score",
               "domain_rating", "domain_citation_rate", "num_lists_tables",
               "has_schema", "has_faq", "num_entities_matched", "word_count"]
by_class = df.groupby(config.TARGET)[signal_cols].mean().T
by_class.columns = ["not cited", "cited"]
by_class["abs_gap"] = (by_class["cited"] - by_class["not cited"]).abs()
by_class.sort_values("abs_gap", ascending=False).round(3)
""")

md("## 4 — Feature engineering\n"
   "`build_xy` adds two domain-motivated features — `rank_reciprocal` "
   "(1/rank, the non-linear visibility decay) and `structure_score` (a single "
   "'how extractable is this page' signal) — and casts categoricals to pandas "
   "`category` dtype so LightGBM handles them natively.")
code(r"""
X, y, groups = build_xy(df)
print("X shape:", X.shape)
print("categoricals:", [c for c in X.columns if str(X[c].dtype) == 'category'])
print("any NaNs in X:", bool(X.isna().any().any()))
X.head(3)
""")

md("## 5 — Leakage-safe cross-validation\n"
   "`GroupKFold(groups=query_id)` keeps every query wholly inside one fold. We "
   "collect **out-of-fold** predictions so the headline metric is genuinely "
   "held-out.")
code(r"""
cv = run_group_kfold_cv(X, y, groups, verbose=True)
print(f"\nPer-fold PR-AUC: {cv.mean_ap:.4f} +/- {cv.std_ap:.4f}")
""")

md("## 6 — Evaluation vs baselines\n"
   "The bar to clear isn't zero — it's the **rank-only heuristic** (predict "
   "citation from organic position alone). If the learned model doesn't beat "
   "that, it isn't earning its keep.")
code(r"""
summary = ev.evaluation_summary(df, cv.oof_pred)

rows = [
    ("PR-AUC (average precision)", summary["model_pr_auc"], summary["rank_only_pr_auc"], summary["positive_rate"]),
    ("ROC-AUC",                    summary["model_roc_auc"], summary["rank_only_roc_auc"], 0.5),
    ("Precision@k (per query)",    summary["precision_at_true_k_model"], summary["precision_at_true_k_rank"], np.nan),
]
comp = pd.DataFrame(rows, columns=["metric", "Gap-Miner", "rank-only", "random / prevalence"]).set_index("metric")
print(comp.round(4).to_string())
print(f"\nLift over rank-only (PR-AUC): +{summary['lift_over_rank']:.3f}")
print(f"Lift over prevalence (PR-AUC): +{summary['lift_over_prevalence']:.3f}")
""")

md("### 6.1 Precision-Recall curve")
code(r"""
fig, ax = ev.plot_pr_curves(df, cv.oof_pred)
plt.show()
""")

md("### 6.2 Confusion matrix at the F1-optimal threshold")
code(r"""
fig, ax = ev.plot_confusion(df, cv.oof_pred, threshold=summary["best_f1_threshold"])
plt.show()
print(f"Operating threshold: {summary['best_f1_threshold']:.3f}  |  F1 = {summary['best_f1']:.3f}")
""")

md("## 7 — Explainability (TreeSHAP)\n"
   "Why does a URL get cited? We train a final model on all rows (at the mean "
   "best CV iteration) and attribute predictions with TreeSHAP.")
code(r"""
mean_best_iter = int(np.mean(cv.best_iterations))
final_model = train_final_model(X, y, n_estimators=mean_best_iter)
explainer, shap_values = ex.compute_shap_values(final_model, X)

importance = ex.mean_abs_importance(shap_values, list(X.columns))
importance.head(10)
""")

md("### 7.1 Global importance and direction (beeswarm)\n"
   "Each dot is a (query, URL) row. Colour = feature value, x-position = push "
   "toward (right) or away from (left) citation.")
code(r"""
ex.plot_beeswarm(shap_values, X)
plt.show()
""")

md("### 7.2 Response curve for the top driver")
code(r"""
top_feature = importance.iloc[0]["feature"]
ex.plot_dependence(shap_values, X, top_feature)
plt.show()
print("Top driver:", top_feature)
""")

md(r"""
## 8 — Read-out & next steps

**What the method shows (on synthetic data).** The learned model beats a strong
rank-only heuristic on PR-AUC and on per-query precision@k, and SHAP attributes
the edge to **semantic passage match, content structure, and domain citation
history** — signals that raw ranking position can't see. That is precisely the
GEO thesis, made measurable: *structured, on-topic pages get cited beyond what
their SERP position alone would predict.*

**From here:**

1. **Real labels.** Swap the synthetic sample for labelled AI Overview citation
   data (same schema) — the OnPagePilot partnership set. No pipeline changes
   needed.
2. **Richer features.** Real embeddings for query↔passage similarity, entity
   coverage from an NER pass, SERP feature flags, page-speed / Core Web Vitals.
3. **Calibration.** `is_unbalance` optimises ranking, not calibrated
   probabilities — add isotonic/Platt calibration if absolute probabilities are
   needed for a "citation likelihood" score.
4. **The product angle.** Per (query, URL) SHAP values become a *gap report*:
   for a page you *don't* own the citation on, SHAP names the specific levers
   (add FAQ schema, tighten the answer passage, raise topical coverage) to close
   the gap. This is the analytical layer a rules-based tool can't offer.
""")

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Wrote {OUT} with {len(cells)} cells")
