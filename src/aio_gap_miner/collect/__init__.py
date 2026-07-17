"""Real-data collection for the AIO Gap-Miner.

Turns a list of queries into labelled (query, URL) training rows in the
Gap-Miner schema, using DataForSEO for SERP + AI Overview citations and an
on-page crawl for content features.

    from aio_gap_miner.collect import build_dataset
    df = build_dataset(["immobilienbewertung münchen", ...])
"""

from __future__ import annotations

from .crawl import extract_onpage_features, fetch_html
from .keywords import expand_seeds, fetch_keyword_suggestions
from .pipeline import (
    authority_features,
    build_dataset,
    collect_query,
    entity_overlap,
    finalise_dataset,
    semantic_scores,
)
from .serp import DataForSEOClient, SerpResult, parse_serp

__all__ = [
    "build_dataset",
    "collect_query",
    "finalise_dataset",
    "DataForSEOClient",
    "SerpResult",
    "parse_serp",
    "fetch_html",
    "extract_onpage_features",
    "semantic_scores",
    "entity_overlap",
    "authority_features",
    "expand_seeds",
    "fetch_keyword_suggestions",
]
