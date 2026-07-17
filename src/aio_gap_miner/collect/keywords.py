"""Keyword expansion via DataForSEO Labs -- real, volume-ranked keyword ideas.

Instead of hand-written guesses, this pulls real long-tail keyword suggestions
(each with real monthly search volume from Google Ads data) from a handful of
seed terms, so the query list feeding the Gap-Miner is itself data-driven --
the same "Universe" step a topical-authority tool like SEOMANTIK performs, but
under your own control and paid at your own DataForSEO cost.

Endpoint (verified against DataForSEO docs, 2026):
``POST https://api.dataforseo.com/v3/dataforseo_labs/google/keyword_suggestions/live``
-- full-text search for long-tail terms containing the seed keyword.
"""

from __future__ import annotations

import requests

from .serp import DataForSEOClient

LABS_KEYWORD_SUGGESTIONS_ENDPOINT = (
    "https://api.dataforseo.com/v3/dataforseo_labs/google/keyword_suggestions/live"
)


def fetch_keyword_suggestions(
    client: DataForSEOClient,
    seed: str,
    location_code: int = 2276,
    language_code: str = "de",
    limit: int = 100,
    min_search_volume: int = 1,
    timeout: int = 60,
) -> list[dict]:
    """Return up to `limit` real long-tail keyword ideas containing `seed`.

    Each item: {"keyword": str, "search_volume": int, "competition": str|None, "cpc": float|None}
    """
    task: dict = {
        "keyword": seed,
        "location_code": location_code,
        "language_code": language_code,
        "limit": limit,
        "order_by": ["keyword_info.search_volume,desc"],
    }
    if min_search_volume > 0:
        task["filters"] = ["keyword_info.search_volume", ">", min_search_volume - 1]

    resp = requests.post(
        LABS_KEYWORD_SUGGESTIONS_ENDPOINT,
        auth=(client.login, client.password),
        json=[task],
        timeout=timeout,
    )
    resp.raise_for_status()
    return _parse_keyword_suggestions(resp.json())


def _parse_keyword_suggestions(raw: dict) -> list[dict]:
    """Extract (keyword, volume, competition, cpc) tuples from the raw response."""
    tasks = raw.get("tasks") or []
    if not tasks:
        return []
    results = tasks[0].get("result") or []
    if not results:
        return []
    items = results[0].get("items") or []

    out = []
    for it in items:
        info = it.get("keyword_info") or {}
        kw = it.get("keyword", "")
        if not kw:
            continue
        out.append(
            {
                "keyword": kw,
                "search_volume": int(info.get("search_volume") or 0),
                "competition": info.get("competition_level"),
                "cpc": info.get("cpc"),
            }
        )
    return out


def expand_seeds(
    client: DataForSEOClient,
    seeds: list[str],
    target_count: int = 300,
    location_code: int = 2276,
    language_code: str = "de",
    per_seed_limit: int = 100,
    verbose: bool = True,
) -> list[str]:
    """Expand seed keywords into up to `target_count` deduped real queries,
    sorted by search volume descending.

    May return fewer than `target_count` if the niche genuinely doesn't have
    that many distinct, volumed long-tail terms -- that's reported, not padded.
    """
    seen: dict[str, int] = {}
    for seed in seeds:
        if verbose:
            print(f"  Seed: {seed!r}")
        items = fetch_keyword_suggestions(
            client,
            seed,
            location_code=location_code,
            language_code=language_code,
            limit=per_seed_limit,
        )
        for it in items:
            kw = it["keyword"].strip()
            if not kw:
                continue
            key = kw.lower()
            vol = it["search_volume"]
            if key not in seen or vol > seen[key][1]:
                seen[key] = (kw, vol)
        if verbose:
            print(f"    -> {len(items)} Ideen, kumuliert {len(seen)} einzigartige")

    ranked = sorted(seen.values(), key=lambda kv: kv[1], reverse=True)
    return [kw for kw, _vol in ranked[:target_count]]
