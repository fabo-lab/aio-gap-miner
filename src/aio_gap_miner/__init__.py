"""AIO Gap-Miner: predicting which URLs get cited in Google AI Overviews.

A supervised, explainable model over (query, URL) pairs. Public API:

    from aio_gap_miner import (
        generate_synthetic_dataset, load_dataset,   # data
        build_xy,                                    # features
        run_group_kfold_cv, train_final_model,       # model
        evaluation_summary,                          # evaluate
        compute_shap_values,                         # explain
    )
"""

from __future__ import annotations

from . import config
from .data import (
    EXPECTED_COLUMNS,
    generate_synthetic_dataset,
    load_dataset,
    save_dataset,
)
from .database import build_database, get_engine, load_candidates, read_sql
from .evaluate import (
    compare_models,
    compute_metrics,
    evaluation_summary,
    precision_at_true_k,
    rank_only_score,
)
from .explain import compute_shap_values, mean_abs_importance
from .features import build_xy, engineer_features
from .model import (
    CVResult,
    run_group_kfold_cv,
    run_logreg_group_kfold_cv,
    train_final_model,
)
from .stats import descriptive_by_class, hypothesis_tests

__version__ = "0.2.0"

__all__ = [
    "config",
    "EXPECTED_COLUMNS",
    "generate_synthetic_dataset",
    "load_dataset",
    "save_dataset",
    "build_database",
    "get_engine",
    "read_sql",
    "load_candidates",
    "engineer_features",
    "build_xy",
    "run_group_kfold_cv",
    "run_logreg_group_kfold_cv",
    "train_final_model",
    "CVResult",
    "compute_metrics",
    "rank_only_score",
    "precision_at_true_k",
    "evaluation_summary",
    "compare_models",
    "compute_shap_values",
    "mean_abs_importance",
    "descriptive_by_class",
    "hypothesis_tests",
    "__version__",
]
