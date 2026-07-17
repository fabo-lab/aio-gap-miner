#!/usr/bin/env python3
"""Run both leakage-safe analysis variants and report them side by side.

This is the honest, real-data analysis. A data audit found that the raw dataset
has label leakage if modelled naively (see ``feature_sets.py`` for the full
write-up). This script reports the two defensible framings:

  Variant A -- ranked pages only (organic_rank kept, rank-101 sentinel rows
               removed): "among pages Google already ranks, which get cited?"
  Variant B -- content signals only (all rows, organic_rank dropped as a
               feature): "what on-page signals distinguish cited pages?"

For each variant it runs LightGBM + Logistic Regression under the *same*
query-grouped GroupKFold CV, compares them against the rank-only heuristic and
the prevalence floor, and writes SHAP importances + plots per variant.

    python scripts/run_variants.py --data data/raw/real.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from aio_gap_miner import config
from aio_gap_miner.data import load_dataset
from aio_gap_miner.evaluate import compare_models
from aio_gap_miner.explain import (
    compute_shap_values,
    mean_abs_importance,
    plot_beeswarm,
    plot_importance_bar,
)
from aio_gap_miner.feature_sets import FEATURE_SETS, prepare_variant
from aio_gap_miner.features import build_xy
from aio_gap_miner.model import (
    run_group_kfold_cv,
    run_logreg_group_kfold_cv,
    train_final_model,
)


def run_one_variant(df_full, variant: str, figures_dir: Path) -> None:
    spec = FEATURE_SETS[variant]
    print("\n" + "=" * 72)
    print(f"  {spec['label']}")
    print("=" * 72)

    df, numeric, categorical = prepare_variant(df_full, variant)
    n_pos = int(df[config.TARGET].sum())
    print(
        f"Rows: {len(df):,}  |  positives: {n_pos:,} ({n_pos / len(df):.1%})  |  "
        f"queries: {df[config.GROUP_COL].nunique()}  |  features: {len(numeric) + len(categorical)}"
    )

    X, y, groups = build_xy(df, numeric_features=numeric, categorical_features=categorical)

    lgbm_cv = run_group_kfold_cv(X, y, groups)
    logreg_oof = run_logreg_group_kfold_cv(X, y, groups)

    scores = {"LightGBM": lgbm_cv.oof_pred, "Logistic Regression": logreg_oof}
    comparison = compare_models(df, scores, groups)
    print("\nModel comparison (per-fold GroupKFold):")
    print(comparison.to_string())

    # SHAP on a final model trained on all rows of this variant.
    model = train_final_model(X, y)
    _explainer, shap_values = compute_shap_values(model, X)
    importance = mean_abs_importance(shap_values, list(X.columns))
    print("\nTop SHAP drivers:")
    print(importance.head(10).to_string(index=False))

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_beeswarm(shap_values, X, save_path=figures_dir / f"shap_beeswarm_variant_{variant}.png")
    plot_importance_bar(
        shap_values, X, save_path=figures_dir / f"shap_importance_variant_{variant}.png"
    )
    print(f"SHAP plots -> {figures_dir}/shap_*_variant_{variant}.png")

    # Persist tables so the results survive outside the terminal (report/Tableau).
    results_dir = config.REPORTS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(results_dir / f"model_comparison_variant_{variant}.csv")
    importance.to_csv(results_dir / f"shap_importance_variant_{variant}.csv", index=False)
    print(f"Tables -> {results_dir}/*_variant_{variant}.csv")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=str, default=None, help="CSV path; defaults to synthetic sample.")
    p.add_argument("--variant", choices=["A", "B", "both"], default="both")
    args = p.parse_args()

    df_full = load_dataset(args.data)
    figures_dir = config.FIGURES_DIR

    variants = ["A", "B"] if args.variant == "both" else [args.variant]
    for v in variants:
        run_one_variant(df_full, v, figures_dir)

    print("\n" + "=" * 72)
    print("Both variants complete. A = ranked-only (rank kept); B = content-only (rank dropped).")
    print(
        "Leaky/dead columns (domain_citation_rate, domain_rating, page_authority) excluded from both."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
