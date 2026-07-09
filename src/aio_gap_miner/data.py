"""Data layer for the AIO Gap-Miner.

The unit of observation is a **(query, URL) pair**: one row per URL that was a
candidate for a given query, labelled ``cited = 1`` if that URL was cited in the
Google AI Overview for the query and ``0`` otherwise.

This module ships a *synthetic* generator so the whole pipeline runs end-to-end
for anyone -- including a reviewer with zero data access. The synthetic labels
are **query-relative**: within each query, the strongest candidates get cited,
which is exactly why leakage-safe (grouped) cross-validation matters downstream.

    >>> from aio_gap_miner.data import generate_synthetic_dataset
    >>> df = generate_synthetic_dataset(n_queries=400)

To train on **real** data, drop a CSV with the same columns (see
``EXPECTED_COLUMNS``) into ``data/raw/`` and point ``load_dataset`` at it. The
real labelling target is: did the URL appear in the AIO citation set for the
query. Feature values come from your SERP scrape + on-page crawl.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Columns every dataset (synthetic or real) is expected to expose.
EXPECTED_COLUMNS: list[str] = [
    "query_id",
    "query",
    "url",
    "organic_rank",
    "domain_rating",
    "page_authority",
    "word_count",
    "has_schema",
    "num_lists_tables",
    "has_faq",
    "query_url_similarity",
    "passage_match_score",
    "content_freshness_days",
    "num_entities_matched",
    "readability_score",
    "is_https",
    "domain_citation_rate",
    "is_forum",
    "is_video",
    "content_type",
    "cited",
]

_CONTENT_TYPES = ["informational", "commercial", "transactional"]

# A small pool of realistic seed queries so the synthetic data reads like a real
# SERP export rather than "query_0001". Purely cosmetic.
_SEED_QUERIES = [
    "how to descale an espresso machine",
    "best home espresso machine 2026",
    "dual boiler vs heat exchanger",
    "what is a verkehrswert",
    "immobilienbewertung kostenlos",
    "how does group kfold work",
    "pr auc vs roc auc",
    "shap values explained",
    "local seo for small business",
    "what is a service area business",
    "google business profile optimization",
    "espresso grind size guide",
    "how to get cited in ai overviews",
    "topical authority explained",
    "entity based seo",
    "schema markup for local business",
]


def _sample_query_features(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Draw ``n`` candidate URLs for a single query with correlated features."""
    # Latent "page quality" that couples authority, ranking and match signals so
    # the synthetic data behaves like the real world (strong pages tend to rank
    # well, match the query, and come from cited domains).
    quality = rng.beta(2.0, 2.5, size=n)

    domain_rating = np.clip(rng.normal(45 + 40 * quality, 12), 1, 100)
    page_authority = np.clip(domain_rating + rng.normal(0, 8, size=n), 1, 100)

    # Better pages rank higher (lower number). Rank is drawn from quality + noise.
    rank_latent = (1 - quality) * 20 + rng.normal(0, 4, size=n)
    organic_rank = np.clip(np.round(rank_latent), 1, 100).astype(int)

    query_url_similarity = np.clip(0.35 + 0.5 * quality + rng.normal(0, 0.12, n), 0, 1)
    passage_match_score = np.clip(query_url_similarity + rng.normal(0, 0.15, n), 0, 1)

    domain_citation_rate = np.clip(
        0.05 + 0.4 * (domain_rating / 100) + rng.normal(0, 0.08, n), 0, 1
    )

    word_count = np.clip(
        rng.normal(1200 + 900 * quality, 600), 150, 6000
    ).round().astype(int)

    has_schema = rng.binomial(1, 0.35 + 0.4 * quality)
    has_faq = rng.binomial(1, 0.20 + 0.35 * quality)
    num_lists_tables = rng.poisson(1 + 4 * quality).clip(0, 20)

    content_freshness_days = np.clip(
        rng.exponential(scale=220) * (1.3 - 0.6 * quality), 1, 2000
    ).round().astype(int)

    num_entities_matched = rng.poisson(3 + 9 * query_url_similarity).clip(0, 40)
    readability_score = np.clip(rng.normal(55 + 15 * quality, 12), 5, 100).round(1)

    is_https = rng.binomial(1, 0.96, size=n)
    is_forum = rng.binomial(1, 0.12, size=n)
    is_video = rng.binomial(1, 0.10, size=n)
    content_type = rng.choice(_CONTENT_TYPES, size=n, p=[0.6, 0.28, 0.12])

    return pd.DataFrame(
        {
            "organic_rank": organic_rank,
            "domain_rating": domain_rating.round(1),
            "page_authority": page_authority.round(1),
            "word_count": word_count,
            "has_schema": has_schema,
            "num_lists_tables": num_lists_tables,
            "has_faq": has_faq,
            "query_url_similarity": query_url_similarity.round(3),
            "passage_match_score": passage_match_score.round(3),
            "content_freshness_days": content_freshness_days,
            "num_entities_matched": num_entities_matched,
            "readability_score": readability_score,
            "is_https": is_https,
            "domain_citation_rate": domain_citation_rate.round(3),
            "is_forum": is_forum,
            "is_video": is_video,
            "content_type": content_type,
        }
    )


def _citation_propensity(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Latent propensity that a candidate URL gets cited (feature-driven + noise).

    The weights encode the empirical intuition from AIO citation research:
    passage/query match and domain citation history dominate, content structure
    (schema, FAQ, lists/tables) and ranking help, and forum sources
    (Reddit-style) get a modest boost. Gaussian noise ensures the label is *not*
    perfectly recoverable from the features -- the model should beat the
    baselines, not hit a perfect score.
    """
    structure = (
        0.4 * df["has_schema"]
        + 0.4 * df["has_faq"]
        + 0.2 * (df["num_lists_tables"].clip(0, 5) / 5)
    )
    rank_reciprocal = 1.0 / df["organic_rank"]

    latent = (
        1.7 * df["query_url_similarity"]
        + 1.5 * df["passage_match_score"]
        + 1.2 * df["domain_citation_rate"]
        + 1.0 * structure
        + 3.0 * rank_reciprocal
        + 0.5 * (df["num_entities_matched"] / 10)
        + 0.4 * df["is_forum"]
        + 0.3 * (df["domain_rating"] / 100)
        - 0.15 * (df["content_freshness_days"] / 365)
    )
    noise = rng.normal(0, 1.1, size=len(df))
    return latent.to_numpy() + noise


def generate_synthetic_dataset(
    n_queries: int = 400,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a synthetic (query, URL) citation dataset.

    Parameters
    ----------
    n_queries:
        Number of distinct queries. Each query gets a variable number of
        candidate URLs (8-29), so the total row count is several thousand.
    seed:
        Seed for the random generator (reproducibility).

    Returns
    -------
    pandas.DataFrame
        One row per (query, URL) pair with all ``EXPECTED_COLUMNS``.
    """
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []

    for q in range(n_queries):
        n_candidates = int(rng.integers(8, 30))
        block = _sample_query_features(rng, n_candidates)

        # Query-relative labelling: score every candidate, then cite the top-k,
        # where k (~1-8) mimics how many sources an AI Overview actually cites.
        propensity = _citation_propensity(block, rng)
        k = int(np.clip(1 + rng.poisson(2.2), 1, min(8, n_candidates)))
        cited_idx = np.argsort(propensity)[::-1][:k]
        cited = np.zeros(n_candidates, dtype=int)
        cited[cited_idx] = 1

        block.insert(0, "url", [f"https://example{q:04d}.com/page-{i}" for i in range(n_candidates)])
        block.insert(0, "query", _SEED_QUERIES[q % len(_SEED_QUERIES)])
        block.insert(0, "query_id", f"q{q:04d}")
        block["cited"] = cited
        frames.append(block)

    df = pd.concat(frames, ignore_index=True)
    return df[EXPECTED_COLUMNS]


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load a citation dataset from CSV.

    If ``path`` is None, the committed synthetic sample is used. Raises a clear
    error if the file is missing or columns don't match the expected schema.
    """
    path = Path(path) if path is not None else config.SAMPLE_DATASET
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Generate the sample with "
            f"`python scripts/generate_sample_data.py`, or point this at your "
            f"own CSV in data/raw/."
        )
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df


def save_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist a dataset to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
