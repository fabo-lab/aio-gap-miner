#!/usr/bin/env python3
"""The generalisation test: does the model work on websites it has never seen?

Why this exists
---------------
An independent review found a fourth leak, and it is the largest one. It is not
in a feature - it is in the cross-validation.

`GroupKFold(query_id)` stops the same *search* appearing on both sides of a
split. It does not stop the same *page* appearing on both sides. And in this
dataset that matters enormously:

  * 4,857 ranked rows contain only 1,361 distinct URLs
  * 509 URLs appear in more than one search, covering 82% of all rows
  * 9 of the 13 content features are effectively constant per URL (they vary in
    0-28 of 1,361 URLs), so the feature vector is close to a page fingerprint

So a page can sit in training and test with an identical feature vector, and the
model can memorise pages and domains instead of learning content patterns.

What that costs, measured:

    grouped by query   PR-AUC 0.548   ROC-AUC 0.740   <- what was published
    grouped by domain  PR-AUC 0.341   ROC-AUC 0.553   <- unseen websites
    rank-only baseline PR-AUC 0.368   ROC-AUC 0.616

On websites it has not seen, the content model does not beat the ranking
heuristic it was supposed to beat.

This does not make the model useless - predicting within known sites is a real
capability, and part of the domain effect is genuine (Google does favour certain
sites). What it invalidates is the *prescriptive* reading: "improve these content
signals and you will get cited" requires exactly the generalisation that breaks.

This script reports every model at every grouping level so the gap is explicit.

    python scripts/test_generalisation.py --data data/raw/real.csv
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

CONTENT = [
    "word_count", "has_schema", "num_lists_tables", "has_faq",
    "query_url_similarity", "passage_match_score", "content_freshness_days",
    "num_entities_matched", "readability_score", "is_https", "is_forum",
    "is_video", "structure_score",
]
# Features that vary within a page depending on the search - the only ones that
# can carry "does this page fit this query" information.
QUERY_DEPENDENT = ["query_url_similarity", "passage_match_score", "num_entities_matched"]
# Features fixed per page - these are what let the model fingerprint a site.
PAGE_CONSTANT = [c for c in CONTENT if c not in QUERY_DEPENDENT]

LGBM = dict(
    objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
    is_unbalance=True, random_state=SEED, verbose=-1,
)


def run_cv(df, y, groups, cols):
    X = df[cols + ["content_type"]].copy()
    X["content_type"] = X["content_type"].astype("category")
    oof = np.zeros(len(X))
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        m = lgb.LGBMClassifier(**LGBM)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return average_precision_score(y, oof), roc_auc_score(y, oof)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/raw/real.csv")
    args = p.parse_args()

    raw = pd.read_csv(args.data)
    df = raw[raw["organic_rank"] != 101].reset_index(drop=True)
    df["structure_score"] = (
        df["has_schema"].fillna(0) + df["has_faq"].fillna(0)
        + (df["num_lists_tables"].fillna(0) > 3).astype(int)
    )
    df["rank_reciprocal"] = 1.0 / df["organic_rank"].clip(lower=1)
    df["domain"] = df["url"].str.extract(r"https?://([^/]+)/", expand=False)
    y = df["cited"].astype(int)

    print("=" * 78)
    print("  GENERALISATION TEST")
    print(f"  {len(df):,} rows | {df['url'].nunique():,} distinct URLs | "
          f"{df['domain'].nunique():,} domains | {df['query_id'].nunique()} queries")
    multi = df.groupby("url").size()
    print(f"  URLs appearing in >1 search: {(multi > 1).sum():,} "
          f"({multi[multi > 1].sum() / len(df):.0%} of rows)")
    print("=" * 78)

    rank_only = 1.0 / df["organic_rank"].clip(lower=1)
    print(f"\n  Baseline - rank-only heuristic:      "
          f"PR-AUC {average_precision_score(y, rank_only):.3f}   "
          f"ROC-AUC {roc_auc_score(y, rank_only):.3f}")
    print(f"  Baseline - random (prevalence):      PR-AUC {y.mean():.3f}   ROC-AUC 0.500")

    groupings = [("query_id", df["query_id"]), ("domain", df["domain"]), ("url", df["url"])]
    feature_sets = [
        ("content only", CONTENT),
        ("content + rank", ["organic_rank", "rank_reciprocal"] + CONTENT),
        ("page-constant features only", PAGE_CONSTANT),
        ("query-dependent features only", QUERY_DEPENDENT),
        ("query-dependent + rank", ["organic_rank", "rank_reciprocal"] + QUERY_DEPENDENT),
    ]

    rows = []
    for fname, cols in feature_sets:
        print(f"\n  {fname}")
        for gname, g in groupings:
            pr, roc = run_cv(df, y, g, cols)
            flag = ""
            if gname == "domain":
                if pr < average_precision_score(y, rank_only):
                    flag = "   <- below the rank-only baseline"
            print(f"    grouped by {gname:9s}  PR-AUC {pr:.3f}   ROC-AUC {roc:.3f}{flag}")
            rows.append({"features": fname, "grouped_by": gname,
                         "pr_auc": round(pr, 3), "roc_auc": round(roc, 3)})

    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "generalisation_test.csv", index=False)

    print("\n" + "=" * 78)
    print("  How to read this")
    print("=" * 78)
    print("  Grouped by query  = the published setting. Pages recur across folds.")
    print("  Grouped by domain = the honest generalisation test: can the model score")
    print("                      a website it has never seen before?")
    print()
    print("  The gap between those two rows is the size of the site-memorisation")
    print("  effect. Report both. The domain-grouped row is the one that supports")
    print("  any prescriptive claim about content.")
    print(f"\n  Table -> {RESULTS}/generalisation_test.csv")


if __name__ == "__main__":
    main()
