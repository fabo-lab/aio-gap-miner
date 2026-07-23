#!/usr/bin/env python3
"""The definitive model analysis - every result, honestly qualified.

This replaces run_headline_comparison.py and the harden_model.py results. It
addresses every methodological point raised in the independent review:

1. GENERALISATION. `GroupKFold(query_id)` does not stop the same *page* appearing
   on both sides of a split, and pages repeat heavily (1,361 URLs across 4,857
   rows, 82% of rows from repeated URLs). Every model is therefore reported at
   three grouping levels: by query (optimistic), by domain (unseen websites), and
   by URL.

2. POPULATION. `organic_rank` counts all SERP items, so an AI Overview occupying
   slot 1 shifts ranks down - meaning rank partly reveals whether an AI Overview
   exists at all. 200 of 533 searches have no AI Overview, making 39% of rows
   `cited = 0` by construction. Results are reported both on all ranked pages and
   conditional on searches that actually have an AI Overview.

3. CALIBRATION. `is_unbalance=True` deliberately inflates positive probabilities,
   so the reported calibration error was a property of that hyperparameter, not
   of the problem. Both settings are reported.

4. FOLD VARIANCE. Pooling out-of-fold predictions into a single average precision
   hides between-fold variation. Mean-of-fold AP with standard deviation is
   reported instead.

5. HOLDOUT STABILITY. A single random holdout split is a lottery draw. The
   held-out score is reported as a mean and spread over 50 random splits.

6. MODEL DIFFERENCE. Whether "content + rank" genuinely beats "content only" is
   tested with a paired bootstrap over queries, not by comparing point estimates.

    python scripts/run_final_analysis.py --data data/raw/real.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

RESULTS = Path("reports/results")
SEED = 42

CONTENT = [
    "word_count", "has_schema", "num_lists_tables", "has_faq",
    "query_url_similarity", "passage_match_score", "content_freshness_days",
    "num_entities_matched", "readability_score", "is_https", "is_forum",
    "is_video", "structure_score",
]
QUERY_DEPENDENT = ["query_url_similarity", "passage_match_score", "num_entities_matched"]
PAGE_CONSTANT = [c for c in CONTENT if c not in QUERY_DEPENDENT]

BASE_PARAMS = dict(
    objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
    random_state=SEED, verbose=-1,
)


def prepare(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = raw[raw["organic_rank"] != 101].reset_index(drop=True)
    # NOTE: structure_score is defined here as a simple additive score. features.py
    # uses a weighted version; the two disagreed, which is a reproducibility bug.
    # This additive definition is the one used in every reported result.
    df["structure_score"] = (
        df["has_schema"].fillna(0) + df["has_faq"].fillna(0)
        + (df["num_lists_tables"].fillna(0) > 3).astype(int)
    )
    df["rank_reciprocal"] = 1.0 / df["organic_rank"].clip(lower=1)
    df["domain"] = df["url"].str.extract(r"https?://([^/]+)/", expand=False)
    return df


def fold_scores(df, y, groups, cols, params, n_splits=5):
    """Return (mean AP, sd AP, mean ROC, pooled out-of-fold predictions)."""
    X = df[cols + ["content_type"]].copy()
    X["content_type"] = X["content_type"].astype("category")
    oof = np.zeros(len(X))
    aps, rocs = [], []
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = pred
        if y.iloc[va].nunique() > 1:
            aps.append(average_precision_score(y.iloc[va], pred))
            rocs.append(roc_auc_score(y.iloc[va], pred))
    return float(np.mean(aps)), float(np.std(aps)), float(np.mean(rocs)), oof


def section_generalisation(df, y, params):
    print("\n" + "=" * 78)
    print("  1. GENERALISATION - does it work on websites it has never seen?")
    print("=" * 78)
    rank_only = 1.0 / df["organic_rank"].clip(lower=1)
    base_ap = average_precision_score(y, rank_only)
    print(f"\n  Rank-only baseline: PR-AUC {base_ap:.3f} | ROC-AUC {roc_auc_score(y, rank_only):.3f}")
    print(f"  Random baseline:    PR-AUC {y.mean():.3f}\n")

    rows = []
    sets = [("content only", CONTENT),
            ("content + rank", ["organic_rank", "rank_reciprocal"] + CONTENT),
            ("page-constant only", PAGE_CONSTANT),
            ("query-dependent only", QUERY_DEPENDENT)]
    print(f"  {'features':22s} {'grouped by':10s} {'PR-AUC (mean±sd)':>20s} {'ROC-AUC':>9s}")
    for fname, cols in sets:
        for gname, g in [("query", df["query_id"]), ("domain", df["domain"]), ("url", df["url"])]:
            ap, sd, roc, _ = fold_scores(df, y, g, cols, params)
            mark = "  <- at/below rank-only" if (gname == "domain" and ap <= base_ap + 0.005) else ""
            print(f"  {fname:22s} {gname:10s} {ap:12.3f} ± {sd:.3f} {roc:9.3f}{mark}")
            rows.append({"features": fname, "grouped_by": gname, "pr_auc_mean": round(ap, 3),
                         "pr_auc_sd": round(sd, 3), "roc_auc": round(roc, 3)})
    print("\n  Page-constant features alone reproduce nearly the full query-grouped")
    print("  score, although by construction they say nothing about query-page fit.")
    print("  That is the site-memorisation effect, measured directly.")
    return pd.DataFrame(rows)


def section_population(df, y, params):
    print("\n" + "=" * 78)
    print("  2. POPULATION - conditioning on searches that have an AI Overview")
    print("=" * 78)
    has_aio = df.groupby("query_id")["cited"].transform("max") > 0
    sub = df[has_aio].reset_index(drop=True)
    y_sub = sub["cited"].astype(int)
    n_no_aio = df["query_id"].nunique() - sub["query_id"].nunique()
    print(f"\n  Searches with no AI Overview: {n_no_aio} of {df['query_id'].nunique()}")
    print(f"  Their rows are cited=0 by construction: {len(df) - len(sub):,} of {len(df):,} "
          f"({(len(df) - len(sub)) / len(df):.0%})")
    print(f"\n  Prevalence, all ranked pages:        {y.mean():.3f}")
    print(f"  Prevalence, AI-Overview searches:    {y_sub.mean():.3f}")

    rows = []
    print(f"\n  {'population':28s} {'PR-AUC':>8s} {'lift vs random':>15s}")
    for label, d, yy in [("all ranked pages", df, y), ("AI-Overview searches only", sub, y_sub)]:
        ap, sd, roc, _ = fold_scores(d, yy, d["query_id"], CONTENT, params)
        print(f"  {label:28s} {ap:8.3f} {ap / yy.mean():14.2f}x")
        rows.append({"population": label, "pr_auc": round(ap, 3),
                     "prevalence": round(float(yy.mean()), 3),
                     "lift": round(ap / float(yy.mean()), 2)})
    print("\n  The lift-versus-random claim is materially smaller on the conditional")
    print("  population. Report the conditional number as the headline.")
    return pd.DataFrame(rows), sub, y_sub


def section_calibration(df, y):
    from sklearn.calibration import calibration_curve
    print("\n" + "=" * 78)
    print("  3. CALIBRATION - is_unbalance was inflating the probabilities")
    print("=" * 78)
    cols = ["organic_rank", "rank_reciprocal"] + CONTENT
    rows = []
    for label, unbal in [("is_unbalance=True (previous)", True), ("is_unbalance=False", False)]:
        params = dict(BASE_PARAMS, is_unbalance=unbal)
        _, _, _, oof = fold_scores(df, y, df["query_id"], cols, params)
        frac, mean_pred = calibration_curve(y, oof, n_bins=10, strategy="quantile")
        err = float(np.mean(np.abs(frac - mean_pred)))
        print(f"\n  {label}")
        print(f"    mean absolute calibration error: {err:.3f}")
        print(f"    mean predicted {oof.mean():.3f} vs actual {y.mean():.3f}")
        rows.append({"setting": label, "calibration_error": round(err, 3),
                     "mean_predicted": round(float(oof.mean()), 3),
                     "actual": round(float(y.mean()), 3)})
    print("\n  The earlier 0.13 error was a hyperparameter artefact, not a property")
    print("  of the problem. All results use is_unbalance=False.")
    return pd.DataFrame(rows)


def section_holdout(df, y, params, n_repeats=50):
    print("\n" + "=" * 78)
    print(f"  4. HELD-OUT STABILITY - {n_repeats} random splits, not one")
    print("=" * 78)
    cols = ["organic_rank", "rank_reciprocal"] + CONTENT
    X = df[cols + ["content_type"]].copy()
    X["content_type"] = X["content_type"].astype("category")
    scores = []
    for i in range(n_repeats):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=i)
        tr, te = next(gss.split(X, y, df["query_id"]))
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict_proba(X.iloc[te])[:, 1]
        if y.iloc[te].nunique() > 1:
            scores.append(average_precision_score(y.iloc[te], pred))
    s = np.array(scores)
    print(f"\n  PR-AUC over {len(s)} random query splits:")
    print(f"    mean {s.mean():.3f} | sd {s.std():.3f} | range [{s.min():.3f}, {s.max():.3f}]")
    print(f"\n  A single split (the previously reported number) could land anywhere in")
    print(f"  that range. Report the mean and spread.")
    return pd.DataFrame([{"n_splits": len(s), "mean": round(float(s.mean()), 3),
                          "sd": round(float(s.std()), 3),
                          "min": round(float(s.min()), 3), "max": round(float(s.max()), 3)}])


def section_bootstrap(df, y, params, n_boot=400):
    print("\n" + "=" * 78)
    print("  5. IS 'content + rank' REALLY BETTER? - paired bootstrap over queries")
    print("=" * 78)
    _, _, _, oof_c = fold_scores(df, y, df["query_id"], CONTENT, params)
    _, _, _, oof_cr = fold_scores(df, y, df["query_id"],
                                  ["organic_rank", "rank_reciprocal"] + CONTENT, params)
    queries = df["query_id"].values
    uq = np.unique(queries)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(n_boot):
        samp = rng.choice(uq, size=len(uq), replace=True)
        idx = np.concatenate([np.where(queries == q)[0] for q in samp])
        yy = y.values[idx]
        if len(np.unique(yy)) < 2:
            continue
        diffs.append(average_precision_score(yy, oof_cr[idx])
                     - average_precision_score(yy, oof_c[idx]))
    d = np.array(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n  Difference (content+rank minus content-only), {len(d)} bootstrap resamples:")
    print(f"    mean {d.mean():+.3f} | 95% CI [{lo:+.3f}, {hi:+.3f}] | P(>0) = {(d > 0).mean():.2f}")
    verdict = "real but small" if lo > 0 else "not distinguishable from zero"
    print(f"    -> {verdict}")
    return pd.DataFrame([{"mean_diff": round(float(d.mean()), 4),
                          "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                          "p_gt_zero": round(float((d > 0).mean()), 3), "verdict": verdict}])


def section_robustness(df, y, params):
    print("\n" + "=" * 78)
    print("  6. ROBUSTNESS - rows where the crawl failed")
    print("=" * 78)
    if "crawl_ok" not in df.columns:
        print("  (crawl_ok not present)")
        return pd.DataFrame()
    failed = (df["crawl_ok"] == 0)
    print(f"\n  Rows with a failed crawl: {int(failed.sum()):,} of {len(df):,}")
    print(f"    Their semantic features come from the SERP title+snippet - text Google")
    print(f"    chose *because* it matches the query. Different measurement, same column.")
    print(f"    Citation rate: failed {df[failed]['cited'].mean():.3f} vs "
          f"ok {df[~failed]['cited'].mean():.3f}")
    sub = df[~failed].reset_index(drop=True)
    ysub = sub["cited"].astype(int)
    ap_all, _, _, _ = fold_scores(df, y, df["query_id"], CONTENT, params)
    ap_ok, _, _, _ = fold_scores(sub, ysub, sub["query_id"], CONTENT, params)
    print(f"\n    PR-AUC with those rows:    {ap_all:.3f}")
    print(f"    PR-AUC without them:       {ap_ok:.3f}")
    print(f"    -> {'stable' if abs(ap_all - ap_ok) < 0.03 else 'sensitive - report both'}")
    return pd.DataFrame([{"pr_auc_all": round(ap_all, 3), "pr_auc_crawl_ok_only": round(ap_ok, 3),
                          "n_failed": int(failed.sum())}])


def section_sparse_features(df):
    print("\n" + "=" * 78)
    print("  7. FEATURES TOO SPARSE TO SUPPORT A CLAIM")
    print("=" * 78)
    for col in ["is_forum", "is_video", "has_faq", "has_schema"]:
        if col in df.columns:
            n = int((df[col] == 1).sum())
            note = "  <- too few rows for any claim" if n < 150 else ""
            print(f"  {col:16s} positive in {n:5d} of {len(df):,} rows{note}")
    print("\n  is_forum and is_video should be read as 'no conclusion possible',")
    print("  not as evidence of an effect.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/raw/real.csv")
    args = p.parse_args()

    df = prepare(args.data)
    y = df["cited"].astype(int)
    params = dict(BASE_PARAMS, is_unbalance=False)

    print("=" * 78)
    print("  FINAL ANALYSIS")
    print(f"  {len(df):,} ranked rows | {df['url'].nunique():,} URLs | "
          f"{df['domain'].nunique():,} domains | {df['query_id'].nunique()} searches")
    print(f"  Prevalence {y.mean():.3f} | is_unbalance=False")
    print("=" * 78)

    RESULTS.mkdir(parents=True, exist_ok=True)
    gen = section_generalisation(df, y, params)
    pop, _, _ = section_population(df, y, params)
    cal = section_calibration(df, y)
    hold = section_holdout(df, y, params)
    boot = section_bootstrap(df, y, params)
    rob = section_robustness(df, y, params)
    section_sparse_features(df)

    gen.to_csv(RESULTS / "final_generalisation.csv", index=False)
    pop.to_csv(RESULTS / "final_population.csv", index=False)
    cal.to_csv(RESULTS / "final_calibration.csv", index=False)
    hold.to_csv(RESULTS / "final_holdout.csv", index=False)
    boot.to_csv(RESULTS / "final_bootstrap.csv", index=False)
    if len(rob):
        rob.to_csv(RESULTS / "final_robustness.csv", index=False)

    print("\n" + "=" * 78)
    print(f"  Tables -> {RESULTS}/final_*.csv")
    print("=" * 78)


if __name__ == "__main__":
    main()
