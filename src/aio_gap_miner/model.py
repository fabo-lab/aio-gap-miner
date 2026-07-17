"""Modeling for the AIO Gap-Miner.

The core routine is leakage-safe cross-validation. Because the label is
*query-relative* (whether a URL is cited depends on the other candidates for the
same query), rows from one query must never straddle the train/test split.
``sklearn.model_selection.GroupKFold`` with ``groups = query_id`` guarantees
that, and the out-of-fold (OOF) predictions it produces give an honest estimate
of held-out performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config


@dataclass
class CVResult:
    """Container for cross-validation outputs."""

    oof_pred: np.ndarray  # out-of-fold predicted probabilities
    fold_ap: list[float] = field(default_factory=list)  # per-fold average precision
    models: list[lgb.LGBMClassifier] = field(default_factory=list)
    best_iterations: list[int] = field(default_factory=list)

    @property
    def mean_ap(self) -> float:
        return float(np.mean(self.fold_ap))

    @property
    def std_ap(self) -> float:
        return float(np.std(self.fold_ap))


def run_group_kfold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    params: dict | None = None,
    n_splits: int = config.N_SPLITS,
    verbose: bool = False,
) -> CVResult:
    """Run GroupKFold CV with LightGBM and collect out-of-fold predictions.

    Parameters
    ----------
    X, y, groups:
        Feature matrix, target, and group labels (see ``features.build_xy``).
    params:
        LightGBM hyperparameters. Defaults to ``config.LGBM_PARAMS``.
    n_splits:
        Number of CV folds.
    verbose:
        If True, print per-fold average precision.

    Returns
    -------
    CVResult
    """
    params = dict(params or config.LGBM_PARAMS)
    from sklearn.metrics import average_precision_score

    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.zeros(len(X), dtype=float)
    result = CVResult(oof_pred=oof_pred)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        proba = model.predict_proba(X_va)[:, 1]
        oof_pred[va_idx] = proba

        ap = average_precision_score(y_va, proba)
        result.fold_ap.append(float(ap))
        result.models.append(model)
        result.best_iterations.append(int(model.best_iteration_ or params["n_estimators"]))

        if verbose:
            print(
                f"  fold {fold + 1}/{n_splits}  PR-AUC = {ap:.4f}  "
                f"(best_iter={model.best_iteration_})"
            )

    return result


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    n_estimators: int | None = None,
) -> lgb.LGBMClassifier:
    """Train a single model on all rows (for SHAP / deployment).

    ``n_estimators`` is typically set to the mean best iteration from CV so the
    final model matches the early-stopped fold models.
    """
    params = dict(params or config.LGBM_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model


def _build_logreg_pipeline(numeric_cols: list[str], categorical_cols: list[str]):
    """Preprocessing + logistic regression: scale numerics, one-hot categoricals.

    A transparent linear classifier -- the foundational "regression &
    classification" baseline the tree model has to beat. Column lists are passed
    in (rather than read from config) so it works with any leakage-safe variant.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("pre", pre),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def run_logreg_group_kfold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = config.N_SPLITS,
) -> np.ndarray:
    """GroupKFold CV for the logistic-regression baseline; returns OOF probs.

    Uses the same query-grouped folds as the main model so the comparison is
    apples-to-apples. Categorical columns are cast to string so the one-hot
    encoder handles them cleanly.
    """
    X_lr = X.copy()
    categorical_cols = [c for c in X_lr.columns if str(X_lr[c].dtype) == "category"]
    numeric_cols = [c for c in X_lr.columns if c not in categorical_cols]
    for col in categorical_cols:
        X_lr[col] = X_lr[col].astype(str)

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(X_lr), dtype=float)
    for tr_idx, va_idx in gkf.split(X_lr, y, groups):
        pipe = _build_logreg_pipeline(numeric_cols, categorical_cols)
        pipe.fit(X_lr.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = pipe.predict_proba(X_lr.iloc[va_idx])[:, 1]
    return oof
