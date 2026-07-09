#!/usr/bin/env python3
"""Export a flat, Tableau-ready extract of the scored dataset.

Produces one wide CSV -- every (query, URL) row with its raw features, the
actual label, both models' predicted probabilities, a "citation gap" flag, and
the single strongest SHAP driver per row. This is the data source for the
Tableau dashboard (see tableau/README.md).

    python scripts/export_tableau.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from aio_gap_miner import config
from aio_gap_miner.data import load_dataset
from aio_gap_miner.evaluate import best_f1_threshold
from aio_gap_miner.explain import compute_shap_values
from aio_gap_miner.features import build_xy
from aio_gap_miner.model import (
    run_group_kfold_cv,
    run_logreg_group_kfold_cv,
    train_final_model,
)

OUT_DIR = config.PROJECT_ROOT / "tableau"
OUT_CSV = OUT_DIR / "aio_gap_miner_tableau.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X, y, groups = build_xy(df)

    print("Scoring with LightGBM (OOF) ...")
    cv = run_group_kfold_cv(X, y, groups)
    print("Scoring with Logistic Regression (OOF) ...")
    oof_lr = run_logreg_group_kfold_cv(X, y, groups)

    thr, _ = best_f1_threshold(y.to_numpy(), cv.oof_pred)

    print("Computing per-row SHAP drivers ...")
    final_model = train_final_model(X, y, n_estimators=int(np.mean(cv.best_iterations)))
    _, shap_values = compute_shap_values(final_model, X)
    feat_names = np.array(X.columns)
    top_idx = np.abs(shap_values).argmax(axis=1)
    top_feature = feat_names[top_idx]
    top_value = shap_values[np.arange(len(shap_values)), top_idx]

    export = df.copy()
    export["pred_proba_lgbm"] = cv.oof_pred.round(4)
    export["pred_proba_logreg"] = oof_lr.round(4)
    export["predicted_cited"] = (cv.oof_pred >= thr).astype(int)
    # Citation gap: model says this URL should be cited, but it isn't (yet) --
    # the actionable opportunity for a page you don't own the citation on.
    export["citation_gap"] = (
        (export["predicted_cited"] == 1) & (export["cited"] == 0)
    ).astype(int)
    export["top_driver"] = top_feature
    export["top_driver_shap"] = top_value.round(4)
    export["decision_threshold"] = round(thr, 4)

    export.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(export):,} rows x {export.shape[1]} cols -> {OUT_CSV}")
    print(f"Citation gaps flagged: {int(export['citation_gap'].sum()):,}")
    print("Top drivers (row-level frequency):")
    print(export["top_driver"].value_counts().head(6).to_string())


if __name__ == "__main__":
    main()
