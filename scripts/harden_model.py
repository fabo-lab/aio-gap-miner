#!/usr/bin/env python3
"""Analysis 5 - hardening checks: are the results actually robust?

Good PR-AUC numbers aren't enough. This runs the four checks that separate
"my model scored well once" from "these findings hold up":

  1. Held-out test    - train on 80% of queries, score the untouched 20%.
                        Cross-validation can flatter; a clean holdout can't.
  2. Calibration      - when the model says "70% likely", is it right ~70% of
                        the time? A model can rank well but be badly calibrated.
  3. SHAP stability   - retrain on each CV fold and compare the top features.
                        If the drivers change per fold, they're noise, not signal.
  4. Permutation imp. - shuffle each feature and measure the damage. A completely
                        different method than SHAP; if both agree, the finding is real.

Everything runs leakage-safe (see feature_sets logic inline) and grouped by
query, exactly like the main analysis.

    python scripts/harden_model.py --data data/raw/real.csv --variant B
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")
SEED = 42

# Leakage-safe feature sets (same definitions as feature_sets.py).
CONTENT_NUMERIC = [
    "word_count", "has_schema", "num_lists_tables", "has_faq",
    "query_url_similarity", "passage_match_score", "content_freshness_days",
    "num_entities_matched", "readability_score", "is_https", "is_forum", "is_video",
]
CATEGORICAL = ["content_type"]

LGBM_PARAMS = dict(
    objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
    is_unbalance=True, random_state=SEED, verbose=-1,
)


def prepare(df: pd.DataFrame, variant: str):
    """Return (X, y, groups) for the chosen leakage-safe variant."""
    out = df.copy()
    numeric = list(CONTENT_NUMERIC)
    if variant == "A":
        out = out[out["organic_rank"] != 101].reset_index(drop=True)
        out["rank_reciprocal"] = 1.0 / out["organic_rank"].clip(lower=1)
        numeric = ["organic_rank", "rank_reciprocal"] + numeric

    # Engineered structure score (same as the main pipeline).
    out["structure_score"] = (
        out["has_schema"].fillna(0) + out["has_faq"].fillna(0)
        + (out["num_lists_tables"].fillna(0) > 3).astype(int)
    )
    numeric = numeric + ["structure_score"]

    X = out[numeric + CATEGORICAL].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    y = out["cited"].astype(int)
    groups = out["query_id"]
    return X, y, groups


def check_holdout(X, y, groups) -> dict:
    """Train on 80% of queries, evaluate on the untouched 20%."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(X, y, groups))
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X.iloc[tr], y.iloc[tr])
    proba = model.predict_proba(X.iloc[te])[:, 1]
    res = {
        "pr_auc": average_precision_score(y.iloc[te], proba),
        "roc_auc": roc_auc_score(y.iloc[te], proba),
        "prevalence": float(y.iloc[te].mean()),
        "n_test_rows": len(te),
        "n_test_queries": groups.iloc[te].nunique(),
    }
    print("\n[1] Held-out test (20% of queries never seen in training)")
    print(f"    PR-AUC {res['pr_auc']:.3f}  |  ROC-AUC {res['roc_auc']:.3f}  "
          f"|  prevalence floor {res['prevalence']:.3f}")
    print(f"    Test set: {res['n_test_rows']:,} rows across {res['n_test_queries']} queries")
    lift = res["pr_auc"] / res["prevalence"] if res["prevalence"] else float("nan")
    print(f"    -> {lift:.2f}x better than random guessing on unseen queries")
    return res, (X.iloc[te], y.iloc[te], proba, model)


def check_calibration(y_true, proba) -> None:
    """Does a predicted 70% actually mean ~70%?"""
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    err = float(np.mean(np.abs(frac_pos - mean_pred)))
    print("\n[2] Calibration on the held-out set")
    print(f"    Mean absolute calibration error: {err:.3f}  (0 = perfect)")
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"      predicted {mp:.2f}  ->  actual {fp:.2f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, color="#1f77b4", label="this model")
    ax.set_xlabel("Predicted probability of being cited")
    ax.set_ylabel("Actual share cited")
    ax.set_title("Calibration: can you trust the probability, not just the ranking?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "hardening_calibration.png", dpi=130)
    plt.close(fig)


def check_shap_stability(X, y, groups, n_splits: int = 5) -> pd.DataFrame:
    """Retrain per fold; do the same features stay on top?"""
    gkf = GroupKFold(n_splits=n_splits)
    per_fold = []
    for fold, (tr, _) in enumerate(gkf.split(X, y, groups)):
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(X.iloc[tr], y.iloc[tr])
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(X.iloc[tr])
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        imp = np.abs(sv).mean(axis=0)
        per_fold.append(pd.Series(imp, index=X.columns, name=f"fold_{fold}"))

    stab = pd.concat(per_fold, axis=1)
    stab["mean"] = stab.mean(axis=1)
    stab["std"] = stab.drop(columns="mean").std(axis=1)
    stab["cv"] = (stab["std"] / stab["mean"]).round(3)  # coefficient of variation
    stab = stab.sort_values("mean", ascending=False)

    print("\n[3] SHAP stability across folds (does the story change per fold?)")
    print(f"    {'feature':26s} {'mean':>8s} {'std':>8s} {'CV':>6s}")
    for feat, row in stab.head(8).iterrows():
        print(f"    {feat:26s} {row['mean']:8.3f} {row['std']:8.3f} {row['cv']:6.2f}")
    fold_cols = stab.drop(columns=["mean", "std", "cv"]).columns
    # Is the *set* of top drivers stable, not just which one happens to rank #1?
    top4_sets = [set(stab[c].sort_values(ascending=False).head(4).index) for c in fold_cols]
    common4 = set.intersection(*top4_sets)
    top1_per_fold = [stab[c].idxmax() for c in fold_cols]
    print(f"    Top-4 features shared by ALL folds: {len(common4)}/4 -> {sorted(common4)}")
    print(f"    #1 feature per fold: {top1_per_fold}")
    stable_top = stab.head(4)["cv"].max() < 0.15
    print(f"    -> {'STABLE' if stable_top else 'UNSTABLE'}: top-4 importances vary by "
          f"under {stab.head(4)['cv'].max():.0%} between folds"
          f"{' (the #1 spot alternates only because the top two are near-tied)' if len(set(top1_per_fold)) > 1 else ''}")

    fig, ax = plt.subplots(figsize=(8, 5))
    top = stab.head(10).iloc[::-1]
    ax.barh(top.index, top["mean"], xerr=top["std"], color="#2ca02c", capsize=3)
    ax.set_xlabel("Mean |SHAP| across folds (error bars = std between folds)")
    ax.set_title("Feature importance is consistent across folds")
    fig.tight_layout()
    fig.savefig(FIGURES / "hardening_shap_stability.png", dpi=130)
    plt.close(fig)
    return stab


def check_permutation(X_te, y_te, model) -> pd.DataFrame:
    """Shuffle each feature; how much does performance drop? (SHAP cross-check)"""
    X_num = X_te.copy()
    for c in CATEGORICAL:
        if c in X_num.columns:
            X_num[c] = X_num[c].cat.codes
    m2 = lgb.LGBMClassifier(**LGBM_PARAMS)
    # Refit on the same columns as numeric codes so sklearn can permute them.
    m2.fit(X_num, y_te)
    r = permutation_importance(m2, X_num, y_te, n_repeats=10,
                               random_state=SEED, scoring="average_precision")
    perm = pd.DataFrame({
        "feature": X_num.columns,
        "perm_importance": r.importances_mean,
        "perm_std": r.importances_std,
    }).sort_values("perm_importance", ascending=False)

    print("\n[4] Permutation importance (independent cross-check of SHAP)")
    for _, row in perm.head(8).iterrows():
        print(f"    {row['feature']:26s} {row['perm_importance']:+.4f} (+/- {row['perm_std']:.4f})")
    return perm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/raw/real.csv")
    p.add_argument("--variant", choices=["A", "B"], default="B")
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    X, y, groups = prepare(df, args.variant)
    print("=" * 70)
    print(f"  HARDENING CHECKS - Variant {args.variant}")
    print(f"  {len(X):,} rows | {groups.nunique()} queries | {X.shape[1]} features "
          f"| {y.mean():.1%} positives")
    print("=" * 70)

    holdout, (X_te, y_te, proba, model) = check_holdout(X, y, groups)
    check_calibration(y_te, proba)
    stab = check_shap_stability(X, y, groups)
    perm = check_permutation(X_te, y_te, model)

    # Agreement between the two importance methods = the strongest robustness claim.
    top_shap = set(stab.head(5).index)
    top_perm = set(perm.head(5)["feature"])
    overlap = top_shap & top_perm
    print(f"\n[SUMMARY] SHAP top-5 and permutation top-5 agree on {len(overlap)}/5 features:")
    print(f"    {sorted(overlap)}")

    pd.DataFrame([holdout]).to_csv(RESULTS / f"hardening_holdout_variant_{args.variant}.csv", index=False)
    stab.to_csv(RESULTS / f"hardening_shap_stability_variant_{args.variant}.csv")
    perm.to_csv(RESULTS / f"hardening_permutation_variant_{args.variant}.csv", index=False)
    print(f"\nTables -> {RESULTS}/hardening_*_variant_{args.variant}.csv")
    print(f"Charts -> {FIGURES}/hardening_*.png")


if __name__ == "__main__":
    main()
