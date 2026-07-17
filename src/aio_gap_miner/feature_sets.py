"""Leakage-safe feature sets for the two analysis framings.

A data audit on the first real dataset surfaced three leakage / dead-weight
issues that must be handled *before* modelling, or every result is an artefact:

1. ``organic_rank == 101`` is a sentinel we assigned to cited URLs that did not
   appear in the organic block. Because such URLs are in the data *only* because
   they were cited, every rank-101 row is ``cited == 1`` -- the feature encodes
   the label. (~27% of rows, ~56% of all positives.)

2. ``domain_rating`` / ``page_authority`` are a constant placeholder (50) by
   design -- no variance, no signal. Harmless but dead; dropped for cleanliness.

3. ``domain_citation_rate`` is computed *from the cited label itself* (share of
   a domain's rows that are cited). For the 572 single-occurrence domains it
   equals ``cited`` exactly -- a direct leak. Dropped from the model here; it can
   be reconstructed leakage-free later via an out-of-fold encoding (the rate for
   a row computed only from *other* folds' labels), which is a hardening-week
   task, not a same-day fix.

Two defensible framings fall out of this, and we report BOTH:

* Variant A -- "Among pages Google already ranks (top ~20 organic), which get
  cited?" Drop the rank-101 rows entirely and model only genuinely ranked URLs.
  Cleanest, easiest to defend; ignores the not-ranked-but-cited pages.

* Variant B -- "What distinguishes cited from non-cited pages by content signals
  alone, independent of rank?" Keep all rows but drop ``organic_rank`` /
  ``rank_reciprocal`` as features (they leak for the 101 group). Closer to the
  Gap-Miner's actual purpose: what you can change on-page to earn a citation.

Both variants also drop the dead authority placeholders and the leaky
``domain_citation_rate``. What remains are genuine, earned content/semantic
signals.
"""

from __future__ import annotations

import pandas as pd

# Features common to both variants: genuine content + semantic + structural
# signals, with all leaky / dead columns removed.
_CONTENT_NUMERIC: list[str] = [
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
    "is_forum",
    "is_video",
]
_CATEGORICAL: list[str] = ["content_type"]

# Variant A additionally keeps rank (legitimate, since 101-rows are removed).
VARIANT_A_NUMERIC: list[str] = ["organic_rank", "rank_reciprocal", *_CONTENT_NUMERIC]
# Variant B drops rank entirely (it leaks for the kept 101-rows).
VARIANT_B_NUMERIC: list[str] = list(_CONTENT_NUMERIC)

FEATURE_SETS: dict[str, dict] = {
    "A": {
        "numeric": VARIANT_A_NUMERIC,
        "categorical": _CATEGORICAL,
        "label": "Variant A - ranked pages only (rank kept, 101-rows removed)",
        "drop_rank_101": True,
    },
    "B": {
        "numeric": VARIANT_B_NUMERIC,
        "categorical": _CATEGORICAL,
        "label": "Variant B - content signals only (all rows, rank dropped)",
        "drop_rank_101": False,
    },
}


def prepare_variant(df: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return ``(df_variant, numeric_features, categorical_features)`` for a variant.

    Variant A removes the rank-101 sentinel rows; variant B keeps every row but
    exposes a feature list without rank. The returned dataframe is a copy.
    """
    if variant not in FEATURE_SETS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {list(FEATURE_SETS)}.")
    spec = FEATURE_SETS[variant]
    out = df.copy()
    if spec["drop_rank_101"]:
        out = out[out["organic_rank"] != 101].reset_index(drop=True)
    return out, list(spec["numeric"]), list(spec["categorical"])
