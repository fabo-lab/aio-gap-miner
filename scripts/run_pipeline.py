#!/usr/bin/env python3
"""Run the full Gap-Miner pipeline end to end and save figures + metrics.

    python scripts/run_pipeline.py                 # uses the synthetic sample
    python scripts/run_pipeline.py --data data/raw/my_real_data.csv

Steps: ETL into SQLite (SQLAlchemy) -> statistics -> features -> GroupKFold CV
(LightGBM + Logistic Regression baselines) -> evaluation -> SHAP. Metrics go to
reports/metrics.json and figures to reports/figures/.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # headless: this CLI only saves figures to disk
import numpy as np

from aio_gap_miner import config
from aio_gap_miner.data import load_dataset
from aio_gap_miner.database import build_database, load_candidates
from aio_gap_miner.evaluate import (
    compare_models,
    evaluation_summary,
    plot_confusion,
    plot_pr_curves,
)
from aio_gap_miner.explain import (
    compute_shap_values,
    mean_abs_importance,
    plot_beeswarm,
    plot_dependence,
    plot_importance_bar,
)
from aio_gap_miner.features import build_xy
from aio_gap_miner.model import (
    run_group_kfold_cv,
    run_logreg_group_kfold_cv,
    train_final_model,
)
from aio_gap_miner.stats import (
    hypothesis_tests,
    plot_correlation_heatmap,
    plot_signal_distributions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None,
                        help="Path to a CSV; defaults to the synthetic sample.")
    args = parser.parse_args()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("1/6  ETL: CSV -> SQLite (SQLAlchemy), read working set back via SQL ...")
    df_csv = load_dataset(args.data)
    engine = build_database(df_csv)
    df = load_candidates(engine)
    print(f"     {len(df):,} rows | {df[config.GROUP_COL].nunique():,} queries "
          f"| {df[config.TARGET].mean():.1%} positive")

    print("2/6  Inferential statistics (cited vs not-cited) ...")
    tests = hypothesis_tests(df)
    print(tests[["feature", "median_cited", "median_not_cited",
                 "p_value", "effect_size_r"]].head(6).to_string(index=False))
    plot_correlation_heatmap(df, save_path=config.FIGURES_DIR / "correlation_heatmap.png")
    plot_signal_distributions(df, save_path=config.FIGURES_DIR / "signal_distributions.png")

    print("3/6  Building features ...")
    X, y, groups = build_xy(df)

    print("4/6  GroupKFold cross-validation ...")
    cv = run_group_kfold_cv(X, y, groups, verbose=True)
    oof_lr = run_logreg_group_kfold_cv(X, y, groups)
    print(f"     LightGBM PR-AUC = {cv.mean_ap:.4f} +/- {cv.std_ap:.4f}")

    print("5/6  Evaluation vs baselines ...")
    comparison = compare_models(df, {
        "Gap-Miner (LightGBM)": cv.oof_pred,
        "Logistic Regression": oof_lr,
    }, groups)
    print(comparison.to_string())

    summary = evaluation_summary(df, cv.oof_pred)
    plot_pr_curves(df, cv.oof_pred, save_path=config.FIGURES_DIR / "pr_curve.png")
    plot_confusion(df, cv.oof_pred, threshold=summary["best_f1_threshold"],
                   save_path=config.FIGURES_DIR / "confusion_matrix.png")

    print("6/6  Final model + SHAP ...")
    final_model = train_final_model(X, y, n_estimators=int(np.mean(cv.best_iterations)))
    _, shap_values = compute_shap_values(final_model, X)
    importance = mean_abs_importance(shap_values, list(X.columns))
    print("     Top drivers (mean |SHAP|):")
    for _, row in importance.head(6).iterrows():
        print(f"       {row['feature']:24s} {row['mean_abs_shap']:.4f}")

    plot_beeswarm(shap_values, X, save_path=config.FIGURES_DIR / "shap_summary.png")
    plot_importance_bar(shap_values, X, save_path=config.FIGURES_DIR / "shap_importance.png")
    plot_dependence(shap_values, X, importance.iloc[0]["feature"],
                    save_path=config.FIGURES_DIR / "shap_dependence.png")

    payload = {
        "cv_pr_auc_mean": cv.mean_ap,
        "cv_pr_auc_std": cv.std_ap,
        "fold_pr_auc": cv.fold_ap,
        "comparison": comparison.reset_index().to_dict(orient="records"),
        **summary,
        "logreg_pr_auc": float(comparison.loc["Logistic Regression", "pr_auc"]),
        "hypothesis_tests": tests.to_dict(orient="records"),
        "top_features": importance.head(10).to_dict(orient="records"),
    }
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"\nDone. Metrics -> {config.REPORTS_DIR / 'metrics.json'}")
    print(f"Figures -> {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
