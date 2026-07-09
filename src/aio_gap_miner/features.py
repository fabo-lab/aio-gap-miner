"""Feature engineering for the AIO Gap-Miner.

Turns a raw (query, URL) DataFrame into the model matrix ``X``, the target
vector ``y``, and the ``groups`` array used for leakage-safe cross-validation.

Two domain-motivated features are engineered here:

* ``rank_reciprocal`` -- ``1 / organic_rank``. Citation propensity falls off
  sharply with rank; the reciprocal captures that non-linearity cleanly.
* ``structure_score`` -- a single "how extractable is this page" signal
  combining schema, FAQ blocks, and list/table density. AI Overviews lift
  answers from structured passages, so this compresses three raw signals into
  one interpretable feature.
"""

from __future__ import annotations

import pandas as pd

from . import config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered feature columns added."""
    out = df.copy()

    # Reciprocal rank: strong, non-linear proxy for above-the-fold visibility.
    out["rank_reciprocal"] = 1.0 / out["organic_rank"].clip(lower=1)

    # Structure score: normalised blend of the three "extractability" signals.
    out["structure_score"] = (
        0.4 * out["has_schema"]
        + 0.4 * out["has_faq"]
        + 0.2 * (out["num_lists_tables"].clip(0, 5) / 5)
    ).round(3)

    return out


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the model matrix, target, and group vector.

    Categorical columns are cast to pandas ``category`` dtype so LightGBM can
    consume them natively (no one-hot encoding).

    Returns
    -------
    (X, y, groups)
        ``X`` -- feature matrix (``config.FEATURES`` columns)
        ``y`` -- target (``config.TARGET``)
        ``groups`` -- group labels (``config.GROUP_COL``) for GroupKFold
    """
    engineered = engineer_features(df)

    X = engineered[config.FEATURES].copy()
    for col in config.CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    y = engineered[config.TARGET].astype(int)
    groups = engineered[config.GROUP_COL]
    return X, y, groups
