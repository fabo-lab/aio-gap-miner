"""Descriptive and inferential statistics for the AIO Gap-Miner.

Before any modelling, we ask a plain statistical question: **do cited and
non-cited URLs actually differ, and on which signals?** This is the A/B-testing
mindset applied to observational data -- treat "cited" vs "not cited" as two
groups and test where they diverge.

Feature distributions here are skewed and non-normal (counts, rates, ranks), so
we use the **Mann-Whitney U** test rather than a Student's t-test, and report a
**rank-biserial effect size** so statistical significance isn't confused with
practical size on ~7k rows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from . import config
from .features import engineer_features

sns.set_theme(style="whitegrid")


def _ensure_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns (rank_reciprocal, structure_score) if missing."""
    if "structure_score" in df.columns and "rank_reciprocal" in df.columns:
        return df
    return engineer_features(df)

# The signals we care about testing (a readable subset of the full feature set).
KEY_SIGNALS: list[str] = [
    "organic_rank",
    "query_url_similarity",
    "passage_match_score",
    "structure_score",
    "domain_rating",
    "domain_citation_rate",
    "num_entities_matched",
    "word_count",
]


def descriptive_by_class(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target: str = config.TARGET,
) -> pd.DataFrame:
    """Mean / median / std of each feature, split by cited vs not cited."""
    features = features or KEY_SIGNALS
    df = _ensure_engineered(df)
    grp = df.groupby(target)[features].agg(["mean", "median", "std"])
    # Reshape to a readable feature-indexed table.
    out = grp.T.unstack(level=-1)
    return out.round(3)


def hypothesis_tests(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target: str = config.TARGET,
) -> pd.DataFrame:
    """Mann-Whitney U test per feature: do cited/non-cited groups differ?

    Returns a table with median difference, U statistic, p-value, and the
    rank-biserial effect size (|r| ~ 0.1 small, 0.3 medium, 0.5 large), sorted
    by effect size descending.
    """
    features = features or KEY_SIGNALS
    df = _ensure_engineered(df)
    cited = df[df[target] == 1]
    other = df[df[target] == 0]

    rows = []
    for f in features:
        a, b = cited[f].to_numpy(), other[f].to_numpy()
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        # Rank-biserial correlation from U (effect size).
        rank_biserial = 1 - (2 * u) / (len(a) * len(b))
        rows.append(
            {
                "feature": f,
                "median_cited": float(np.median(a)),
                "median_not_cited": float(np.median(b)),
                "u_statistic": float(u),
                "p_value": float(p),
                "effect_size_r": float(rank_biserial),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_effect"] = out["effect_size_r"].abs()
    return out.sort_values("abs_effect", ascending=False).drop(columns="abs_effect").reset_index(drop=True)


def correlation_matrix(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Pearson correlation matrix over the numeric feature set."""
    features = features or config.NUMERIC_FEATURES
    df = _ensure_engineered(df)
    return df[features].corr(numeric_only=True)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: list[str] | None = None,
    save_path: str | Path | None = None,
):
    """Seaborn heatmap of the feature correlation matrix."""
    corr = correlation_matrix(df, features)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False, square=True,
                linewidths=0.4, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Feature correlation matrix", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, ax


def plot_signal_distributions(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target: str = config.TARGET,
    save_path: str | Path | None = None,
):
    """Seaborn KDE/hist of key signals, split by citation class."""
    features = (features or KEY_SIGNALS)[:6]
    df = _ensure_engineered(df)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    palette = {0: "#7a8b8b", 1: "#d5602e"}
    for ax, f in zip(axes.ravel(), features):
        sns.kdeplot(data=df, x=f, hue=target, common_norm=False, fill=True,
                    alpha=0.4, palette=palette, ax=ax, legend=(ax is axes.ravel()[0]))
        ax.set_title(f)
        ax.set_xlabel("")
    fig.suptitle("Signal distributions: cited (orange) vs not cited (grey)", y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, axes
