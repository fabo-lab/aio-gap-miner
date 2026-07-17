"""Assemble real (query, URL) training rows in the Gap-Miner schema.

Ties together the SERP client (labels + rank), the crawler (on-page features),
semantic similarity, authority, and entity overlap, and emits a DataFrame with
exactly ``config.EXPECTED_COLUMNS`` -- the same schema the synthetic sample
uses, so ``run_pipeline.py --data <this.csv>`` just works.

Two features are computed *across* the whole collected set, not per row:
``domain_citation_rate`` (how often a domain's candidates get cited) and the
imputation of any missing on-page values.

Semantic similarity uses TF-IDF by default (zero heavy dependencies). If
``sentence-transformers`` is installed it is used instead for true embeddings --
set that up for the strongest ``query_url_similarity`` signal.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..data import EXPECTED_COLUMNS
from .crawl import PageContent, extract_onpage_features, fetch_html
from .serp import SERP_FEATURE_TYPES, Candidate, DataForSEOClient, SerpResult, load_serp_fixture

# --------------------------------------------------------------------------- #
# Semantic similarity (TF-IDF default, sentence-transformers if available)
# --------------------------------------------------------------------------- #
_ST_MODEL = None
_ST_TRIED = False


def _get_st_model():
    """Lazily load a sentence-transformers model on first use (if installed)."""
    global _ST_MODEL, _ST_TRIED
    if _ST_TRIED:
        return _ST_MODEL
    _ST_TRIED = True
    try:
        from sentence_transformers import SentenceTransformer

        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _ST_MODEL = None
    return _ST_MODEL


def _split_passages(text: str, max_passages: int = 40) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    parts = [p.strip() for p in parts if len(p.split()) >= 5]
    return parts[:max_passages] or [text[:500]]


def semantic_scores(query: str, page_text: str) -> tuple[float, float]:
    """Return (query_url_similarity, passage_match_score), both in [0, 1]."""
    if not page_text.strip():
        return 0.0, 0.0

    passages = _split_passages(page_text)
    model = _get_st_model()

    if model is not None:
        q = model.encode([query], normalize_embeddings=True)
        docs = model.encode([page_text[:2000]] + passages, normalize_embeddings=True)
        whole = float(np.dot(q[0], docs[0]))
        best = float(max(np.dot(q[0], d) for d in docs[1:]))
        return max(0.0, whole), max(0.0, best)

    # TF-IDF fallback: fit on the query + this page's passages.
    corpus = [query, page_text[:5000], *passages]
    try:
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(corpus)
        sims = cosine_similarity(tfidf[0], tfidf[1:]).ravel()
    except ValueError:
        return 0.0, 0.0
    whole = float(sims[0]) if len(sims) else 0.0
    best = float(sims[1:].max()) if len(sims) > 1 else whole
    return whole, best


# --------------------------------------------------------------------------- #
# Entity overlap (lightweight; spaCy NER is the documented upgrade)
# --------------------------------------------------------------------------- #
def entity_overlap(query: str, page_text: str) -> int:
    """How many distinctive query terms appear in the page (proxy for entities)."""
    terms = {w for w in re.findall(r"[A-Za-zÄÖÜäöüß]{4,}", query.lower())}
    if not terms or not page_text:
        return 0
    page = page_text.lower()
    return int(sum(1 for t in terms if t in page))


# --------------------------------------------------------------------------- #
# Authority (Moz if configured, neutral fallback otherwise)
# --------------------------------------------------------------------------- #
def authority_features(domain: str) -> tuple[float, float]:
    """Return (domain_rating, page_authority) in [0, 100].

    Deliberately NOT wired to a real backlink-based authority score (Moz,
    Ahrefs, DataForSEO Backlinks bulk_rank, etc.). These proxies are estimates
    built on backlink-crawl coverage that varies significantly by country and
    vertical -- for a DACH/niche market they'd be noisy at best. Rather than
    present a third-party heuristic as if it were ground truth, this pipeline
    leans on ``domain_citation_rate`` instead: the domain's own empirical
    citation track record in the collected data, measured directly against the
    actual target rather than approximated via a generic link-graph score.

    A constant placeholder costs nothing: with zero variance across rows, both
    the model and SHAP correctly assign it ~0 importance -- it doesn't bias
    results, it simply contributes no signal, same as omitting it.
    """
    return 50.0, 50.0  # neutral placeholder, by design -- see docstring above


# --------------------------------------------------------------------------- #
# Row assembly
# --------------------------------------------------------------------------- #
def _row_from_candidate(query_id: str, query: str, cand: Candidate, page: PageContent) -> dict:
    onpage = extract_onpage_features(page, title_hint=cand.title)
    text_for_semantics = onpage.pop("page_text") or f"{cand.title} {cand.snippet}"
    sim, passage = semantic_scores(query, text_for_semantics)
    dr, pa = authority_features(cand.domain)

    return {
        "query_id": query_id,
        "query": query,
        "url": cand.url,
        "organic_rank": int(cand.rank_absolute),
        "domain_rating": dr,
        "page_authority": pa,
        "word_count": onpage["word_count"],
        "has_schema": onpage["has_schema"],
        "num_lists_tables": onpage["num_lists_tables"],
        "has_faq": onpage["has_faq"],
        "query_url_similarity": round(sim, 3),
        "passage_match_score": round(passage, 3),
        "content_freshness_days": onpage["content_freshness_days"],
        "num_entities_matched": entity_overlap(query, text_for_semantics),
        "readability_score": onpage["readability_score"],
        "is_https": onpage["is_https"],
        "domain_citation_rate": np.nan,  # filled in globally below
        "is_forum": onpage["is_forum"],
        "is_video": onpage["is_video"],
        "content_type": onpage["content_type"],
        "cited": int(cand.cited),
        # Extra reference columns (beyond EXPECTED_COLUMNS): not used by the ML
        # pipeline (build_xy selects only config.FEATURES), kept so you can
        # manually spot-check rows or re-derive different features later
        # without re-crawling.
        "title": cand.title,
        "snippet": cand.snippet,
        "crawl_ok": int(page.ok),
    }


def finalise_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-row features and impute missing on-page values.

    Public entry point (also used by the CLI for incremental collection: raw
    rows are appended to disk per-query as a safety net, then this is applied
    once at the end over the full accumulated set, since domain_citation_rate
    is a global aggregate that needs every row to be computed correctly).
    """
    # Domain-level historical citation rate, computed from this dataset.
    domain = df["url"].str.extract(r"https?://([^/]+)/?", expand=False).fillna("")
    rate = df.assign(_d=domain).groupby("_d")["cited"].transform("mean")
    df["domain_citation_rate"] = rate.round(3)

    # Impute on-page features that failed to crawl.
    for col, default in [
        ("word_count", df["word_count"].median()),
        ("has_schema", 0),
        ("has_faq", 0),
        ("num_lists_tables", df["num_lists_tables"].median()),
        ("readability_score", df["readability_score"].median()),
        ("content_freshness_days", df["content_freshness_days"].median()),
    ]:
        fill = 0 if pd.isna(default) else default
        df[col] = df[col].fillna(fill)

    # Keep EXPECTED_COLUMNS first (stable schema for the ML pipeline), plus any
    # extra reference columns (title, snippet, crawl_ok) after -- build_xy()
    # only ever selects config.FEATURES, so extras are inert for modelling but
    # preserved here for manual review / future re-derivation without re-crawling.
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    return df[EXPECTED_COLUMNS + extra_cols]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def collect_query(
    query: str,
    query_id: str,
    client: DataForSEOClient | None,
    *,
    location_code: int,
    language_code: str,
    max_organic: int,
    crawl: bool,
    fixture: str | None,
    polite_delay: float,
    cache_dir: Path | None = None,
) -> tuple[list[dict], dict]:
    """Collect all candidate rows for one query (live API or fixture).

    Returns ``(rows, query_meta)`` -- ``rows`` are (query, URL) candidate rows
    as before; ``query_meta`` is one row of query-level intent signals (SERP
    feature types present, AI Overview stats) for clustering queries by intent
    (local vs informational vs transactional).

    If ``cache_dir`` is given, the raw DataForSEO JSON response and every
    crawled page's raw HTML are additionally saved under it (``serp_json/`` and
    ``html/``) -- a permanent local snapshot so future re-analysis (different
    features, different questions) never needs a re-query or a re-crawl.
    """
    if fixture is not None:
        serp: SerpResult = load_serp_fixture(fixture, query)
    else:
        raw = client.fetch_serp(query, location_code, language_code)
        if cache_dir is not None:
            try:
                serp_dir = cache_dir / "serp_json"
                serp_dir.mkdir(parents=True, exist_ok=True)
                (serp_dir / f"{query_id}.json").write_text(
                    json.dumps(raw, ensure_ascii=False), encoding="utf-8"
                )
            except OSError:
                pass  # caching is a bonus, never let it break the actual collection
        from .serp import parse_serp

        serp = parse_serp(raw, query)

    # Keep top-N organic candidates plus every cited URL.
    ranked = sorted(serp.candidates, key=lambda c: c.rank_absolute)
    kept: list[Candidate] = []
    organic_seen = 0
    for c in ranked:
        if c.cited or organic_seen < max_organic:
            kept.append(c)
            if not c.cited:
                organic_seen += 1

    rows = []
    for i, c in enumerate(kept):
        html_cache_path = (
            cache_dir / "html" / f"{query_id}__{i:02d}.html" if cache_dir is not None else None
        )
        page = (
            fetch_html(c.url, cache_path=html_cache_path)
            if crawl
            else PageContent(url=c.url, ok=False)
        )
        rows.append(_row_from_candidate(query_id, query, c, page))
        if crawl and polite_delay:
            time.sleep(polite_delay)

    query_meta = {
        "query_id": query_id,
        "query": query,
        "num_candidates": len(serp.candidates),
        "num_organic_results": serp.num_organic_results,
        "num_cited": sum(1 for c in serp.candidates if c.cited),
        "ai_overview_present": int(serp.aio_present),
        "ai_overview_num_references": serp.aio_num_references,
        "ai_overview_is_async": int(serp.aio_is_async),
        **{f"has_{feat}": int(feat in serp.serp_features) for feat in SERP_FEATURE_TYPES},
    }
    return rows, query_meta


def build_dataset(
    queries: list[str],
    *,
    location_code: int = 2276,
    language_code: str = "de",
    max_organic: int = 15,
    crawl: bool = True,
    fixture: str | None = None,
    polite_delay: float = 1.0,
    verbose: bool = True,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect a full (query, URL) dataset in the Gap-Miner schema.

    Pass ``fixture=<path>`` to run offline against a saved SERP JSON (no API,
    no credentials) -- used by ``--dry-run``. Otherwise DataForSEO credentials
    must be available in the environment.

    Returns
    -------
    (candidates_df, query_meta_df)
        ``candidates_df`` -- one row per (query, URL) pair, the ML training set.
        ``query_meta_df`` -- one row per query: SERP feature flags (local_pack,
        people_also_ask, etc.) and AI Overview stats, for clustering queries by
        intent (local vs informational vs transactional).
    """
    client = None if fixture is not None else DataForSEOClient()

    all_rows: list[dict] = []
    all_meta: list[dict] = []
    for i, q in enumerate(queries):
        qid = f"q{i:04d}"
        if verbose:
            print(f"[{i + 1}/{len(queries)}] {q!r}")
        rows, meta = collect_query(
            q,
            qid,
            client,
            location_code=location_code,
            language_code=language_code,
            max_organic=max_organic,
            crawl=crawl,
            fixture=fixture,
            polite_delay=polite_delay,
            cache_dir=cache_dir,
        )
        all_rows.extend(rows)
        all_meta.append(meta)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No rows collected -- check queries / credentials / AIO presence.")
    meta_df = pd.DataFrame(all_meta)
    return finalise_dataset(df), meta_df
