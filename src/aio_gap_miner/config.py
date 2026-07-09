"""Central configuration for the AIO Gap-Miner.

Everything the pipeline needs to agree on -- paths, the exact feature set,
the target/group columns, and model hyperparameters -- lives here so the
notebook, the CLI pipeline, and the tests all read from a single source of
truth.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Resolve the project root two levels up from this file (src/aio_gap_miner/).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SAMPLE_DIR: Path = DATA_DIR / "sample"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

SAMPLE_DATASET: Path = SAMPLE_DIR / "aio_citations_sample.csv"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- #
# Schema: the unit of observation is a (query, URL) pair.
# --------------------------------------------------------------------------- #
TARGET: str = "cited"  # 1 if the URL was cited in the AI Overview, else 0
GROUP_COL: str = "query_id"  # rows are grouped by query for leakage-safe CV

# Numeric features fed directly to the model. Some are raw signals from the
# SERP / crawl; a few are engineered in features.py (rank_reciprocal,
# structure_score).
NUMERIC_FEATURES: list[str] = [
    "organic_rank",
    "rank_reciprocal",  # engineered: 1 / organic_rank
    "domain_rating",
    "page_authority",
    "word_count",
    "has_schema",
    "num_lists_tables",
    "has_faq",
    "structure_score",  # engineered: schema + faq + list/table density
    "query_url_similarity",
    "passage_match_score",
    "content_freshness_days",
    "num_entities_matched",
    "readability_score",
    "is_https",
    "domain_citation_rate",
    "is_forum",
    "is_video",
]

# Categorical features. LightGBM consumes these natively via pandas'
# ``category`` dtype -- no one-hot encoding required.
CATEGORICAL_FEATURES: list[str] = [
    "content_type",
]

FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
N_SPLITS: int = 5

# --------------------------------------------------------------------------- #
# LightGBM hyperparameters (binary classification, imbalance-aware).
# `is_unbalance=True` lets LightGBM reweight the rare positive class; a modest
# tree depth and strong regularisation keep it honest on a small dataset.
# --------------------------------------------------------------------------- #
LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.2,
    "is_unbalance": True,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# Early-stopping patience used during cross-validation.
EARLY_STOPPING_ROUNDS: int = 50
