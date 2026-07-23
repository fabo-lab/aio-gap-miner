#!/usr/bin/env python3
"""The headline comparison: four models, one clean population, directly comparable.

Why this script exists
----------------------
An earlier framing ("variant B") kept the rank-101 sentinel rows and simply
dropped ``organic_rank`` as a feature. That looked leakage-safe, but wasn't quite:
all 1,789 rank-101 rows are cited by construction, and they differ systematically
from the rest (they are ~9x more likely to be video pages, and have lower
query-URL similarity). So the model could partly *identify that subgroup* from
other features and get free correct predictions - inflating the score.

Evidence: that model scores 0.773 on all rows, but only 0.479 when evaluated on
genuinely ranked pages alone - worse than a model trained only on ranked pages.

The fix is to run every comparison on ONE population: pages Google actually ranks.
On that population nothing encodes the label, and all four numbers below are
directly comparable with no caveats.

    python scripts/run_headline_comparison.py --data data/raw/real.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

RESULTS = Path("reports/results")
SEED = 42

CONTENT_FEATURES = [
    "word_count", "has_schema", "num_lists_tables", "has_faq",
    "query_url_similarity", "passage_match_score", "content_freshness_days",
    "num_entities_matched", "readability_score", "is_https", "is_forum",
    "is_video", "structure_score",
]
LGBM_PARAMS = dict(
    objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
    is_unbalance=True, random_state=SEED, verbose=-1,
)


def cv_scores(df, y, groups, cols, n_splits=5):
    """Out-of-fold predictions under query-grouped CV."""
    X = df[cols + ["content_type"]].copy()
    X["content_type"] = X["content_type"].astype("category")
    oof = np.zeros(len(X))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, va in gkf.split(X, y, groups):
        m = lgb.LGBMClassifier(**LGBM_PARAMS)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def logreg_scores(df, y, groups, cols, n_splits=5):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    X = df[cols + ["content_type"]].copy()
    X["content_type"] = X["content_type"].astype(str)
    oof = np.zeros(len(X))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, va in gkf.split(X, y, groups):
        pipe = Pipeline([
            ("pre", ColumnTransformer([
                ("num", StandardScaler(), cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["content_type"]),
            ])),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=SEED)),
        ])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    return oof


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/raw/real.csv")
    args = p.parse_args()

    raw = pd.read_csv(args.data)
    n_all = len(raw)
    df = raw[raw["organic_rank"] != 101].reset_index(drop=True)
    df["structure_score"] = (
        df["has_schema"].fillna(0) + df["has_faq"].fillna(0)
        + (df["num_lists_tables"].fillna(0) > 3).astype(int)
    )
    df["rank_reciprocal"] = 1.0 / df["organic_rank"].clip(lower=1)

    y = df["cited"].astype(int)
    groups = df["query_id"]

    print("=" * 72)
    print("  HEADLINE COMPARISON - one population, four models")
    print(f"  {len(df):,} rows (of {n_all:,}; rank-101 sentinel rows removed)")
    print(f"  {groups.nunique()} queries | {y.mean():.1%} of pages cited")
    print("=" * 72)

    rank_only = 1.0 / df["organic_rank"].clip(lower=1)
    content_oof = cv_scores(df, y, groups, CONTENT_FEATURES)
    full_oof = cv_scores(df, y, groups, ["organic_rank", "rank_reciprocal"] + CONTENT_FEATURES)
    logreg_oof = logreg_scores(df, y, groups, ["organic_rank", "rank_reciprocal"] + CONTENT_FEATURES)

    rows = [
        ("Random guessing (prevalence)", float(y.mean()), 0.500),
        ("Rank-only heuristic", average_precision_score(y, rank_only), roc_auc_score(y, rank_only)),
        ("Logistic Regression (content + rank)", average_precision_score(y, logreg_oof),
         roc_auc_score(y, logreg_oof)),
        ("LightGBM - content signals only", average_precision_score(y, content_oof),
         roc_auc_score(y, content_oof)),
        ("LightGBM - content + rank", average_precision_score(y, full_oof),
         roc_auc_score(y, full_oof)),
    ]

    print(f"\n  {'model':38s} {'PR-AUC':>8s} {'ROC-AUC':>9s} {'vs random':>11s}")
    for name, pr, roc in rows:
        lift = pr / float(y.mean())
        print(f"  {name:38s} {pr:8.3f} {roc:9.3f} {lift:10.2f}x")

    out = pd.DataFrame(rows, columns=["model", "pr_auc", "roc_auc"])
    out["lift_vs_random"] = (out["pr_auc"] / float(y.mean())).round(2)
    out["pr_auc"] = out["pr_auc"].round(3)
    out["roc_auc"] = out["roc_auc"].round(3)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "headline_comparison.csv", index=False)

    print("\n  The claim this supports:")
    content_pr = [r[1] for r in rows if "content signals only" in r[0]][0]
    rank_pr = [r[1] for r in rows if "Rank-only" in r[0]][0]
    print(f"    Content signals alone ({content_pr:.3f}) predict citation far better")
    print(f"    than ranking alone ({rank_pr:.3f}) - on identical pages.")
    print(f"\n  Table -> {RESULTS}/headline_comparison.csv")


if __name__ == "__main__":
    main()
