"""Sanity tests for the data + feature layer.

These are deliberately lightweight guards against the mistakes that quietly
break an ML pipeline: schema drift, NaNs sneaking into the matrix, and group
leakage between the label and the CV grouping key.
"""

from __future__ import annotations

import numpy as np

from aio_gap_miner import config
from aio_gap_miner.data import EXPECTED_COLUMNS, generate_synthetic_dataset
from aio_gap_miner.features import build_xy, engineer_features


def test_synthetic_schema():
    df = generate_synthetic_dataset(n_queries=50, seed=0)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) > 0
    # Every query cites at least one URL (AI Overviews cite something).
    per_query = df.groupby("query_id")["cited"].sum()
    assert (per_query >= 1).all()


def test_positive_rate_is_realistic():
    df = generate_synthetic_dataset(n_queries=300, seed=0)
    rate = df["cited"].mean()
    # Rare but not vanishing: this is the regime where PR-AUC is the right metric.
    assert 0.05 < rate < 0.35


def test_engineered_features_present():
    df = generate_synthetic_dataset(n_queries=20, seed=1)
    eng = engineer_features(df)
    assert "rank_reciprocal" in eng.columns
    assert "structure_score" in eng.columns
    assert (eng["rank_reciprocal"] > 0).all()


def test_build_xy_shapes_and_no_nans():
    df = generate_synthetic_dataset(n_queries=40, seed=2)
    X, y, groups = build_xy(df)
    assert list(X.columns) == config.FEATURES
    assert len(X) == len(y) == len(groups)
    assert not X.isna().any().any(), "feature matrix contains NaNs"
    assert set(np.unique(y)).issubset({0, 1})


def test_categoricals_are_category_dtype():
    df = generate_synthetic_dataset(n_queries=20, seed=3)
    X, _, _ = build_xy(df)
    for col in config.CATEGORICAL_FEATURES:
        assert str(X[col].dtype) == "category"
