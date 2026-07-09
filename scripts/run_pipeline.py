#!/usr/bin/env python3
"""Run the full Gap-Miner pipeline end to end and save figures + metrics.

    python scripts/run_pipeline.py                 # uses the synthetic sample
    python scripts/run_pipeline.py --data data/raw/my_real_data.csv

Outputs: metrics to stdout (+ reports/metrics.json) and figures to
reports/figures/.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # headless: this CLI only saves figures to disk
import numpy as np

from aio_gap_miner import config
from aio_gap_miner.data import load_dataset
from aio_gap_miner.evaluate import evaluation_summary, plot_confusion, plot_pr_curves
from aio_gap_miner.explain import (
    compute_shap_values,
    mean_abs_importance,
    plot_beeswarm,
    plot_dependence,
    plot_importance_bar,
)
from aio_gap_miner.features import build_xy
from aio_gap_miner.model import run_group_kfold_cv, train_final_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None,
                        help="Path to a CSV; defaults to the synthetic sample.")
    args = parser.parse_args()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("1/5  Loading data ...")
    df = load_dataset(args.data)
    print(f"     {len(df):,} rows | {df[config.GROUP_COL].nunique():,} queries "
          f"| {df[config.TARGET].mean():.1%} positive")

    print("2/5  Building features ...")
    X, y, groups = build_xy(df)

    print("3/5  GroupKFold cross-validation (LightGBM) ...")
    cv = run_group_kfold_cv(X, y, groups, verbose=True)
    print(f"     PR-AUC = {cv.mean_ap:.4f} +/- {cv.std_ap:.4f}")

    print("4/5  Evaluating vs baselines ...")
    summary = evaluation_summary(df, cv.oof_pred)
    for k, v in summary.items():
        print(f"     {k:28s}: {v:.4f}" if isinstance(v, float) else f"     {k:28s}: {v}")

    plot_pr_curves(df, cv.oof_pred, save_path=config.FIGURES_DIR / "pr_curve.png")
    plot_confusion(df, cv.oof_pred, threshold=summary["best_f1_threshold"],
                   save_path=config.FIGURES_DIR / "confusion_matrix.png")

    print("5/5  Training final model + SHAP ...")
    mean_best_iter = int(np.mean(cv.best_iterations))
    final_model = train_final_model(X, y, n_estimators=mean_best_iter)
    _, shap_values = compute_shap_values(final_model, X)

    importance = mean_abs_importance(shap_values, list(X.columns))
    print("     Top drivers (mean |SHAP|):")
    for _, row in importance.head(6).iterrows():
        print(f"       {row['feature']:24s} {row['mean_abs_shap']:.4f}")

    plot_beeswarm(shap_values, X, save_path=config.FIGURES_DIR / "shap_summary.png")
    plot_importance_bar(shap_values, X, save_path=config.FIGURES_DIR / "shap_importance.png")
    top_feature = importance.iloc[0]["feature"]
    plot_dependence(shap_values, X, top_feature,
                    save_path=config.FIGURES_DIR / "shap_dependence.png")

    metrics_path = config.REPORTS_DIR / "metrics.json"
    payload = {
        "cv_pr_auc_mean": cv.mean_ap,
        "cv_pr_auc_std": cv.std_ap,
        "fold_pr_auc": cv.fold_ap,
        **summary,
        "top_features": importance.head(10).to_dict(orient="records"),
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"\nDone. Metrics -> {metrics_path}")
    print(f"Figures -> {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
