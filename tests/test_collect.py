"""Offline tests for the real-data collection pipeline.

No network required: SERP parsing runs against the bundled fixture, on-page
extraction runs against an inline HTML string, and the full assembly runs in
dry-run mode. These lock down the parts that don't need credentials -- the
label logic, feature extraction, and schema conformance.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from aio_gap_miner.collect.crawl import PageContent, extract_onpage_features, fetch_html
from aio_gap_miner.collect.pipeline import (
    build_dataset,
    collect_query,
    entity_overlap,
    semantic_scores,
)
from aio_gap_miner.collect.serp import load_serp_fixture, normalize_url
from aio_gap_miner.data import EXPECTED_COLUMNS

FIXTURE = Path(__file__).parent / "fixtures" / "serp_sample.json"

SAMPLE_HTML = """
<html><head>
<title>The Complete Descaling Guide</title>
<meta property="article:modified_time" content="2026-05-01T10:00:00Z">
<script type="application/ld+json">
{"@type": "FAQPage", "mainEntity": []}
</script>
</head><body>
<h1>How to descale your espresso machine</h1>
<p>Descaling removes mineral buildup from your machine. Do it every two months.</p>
<ul><li>Step one</li><li>Step two</li></ul>
<table><tr><td>Water hardness</td><td>Frequency</td></tr></table>
<details><summary>How often?</summary><p>Every 2-3 months.</p></details>
</body></html>
"""


def test_parse_serp_labels():
    serp = load_serp_fixture(FIXTURE, "how to descale an espresso machine")
    assert serp.aio_present is True
    cited = serp.cited_urls
    # The three AIO references are labelled cited.
    assert normalize_url("https://www.home-barista.com/descaling-guide") in cited
    assert normalize_url("https://www.wholelattelove.com/blogs/articles/how-to-descale") in cited
    assert normalize_url("https://www.reddit.com/r/espresso/comments/descaling") in cited
    # An organic result that is NOT cited stays label 0.
    yt = next(c for c in serp.candidates if "youtube" in c.url)
    assert yt.cited is False


def test_cited_not_in_organic_gets_rank_sentinel():
    serp = load_serp_fixture(FIXTURE, "q")
    reddit = next(c for c in serp.candidates if "reddit" in c.url)
    assert reddit.cited is True
    assert reddit.rank_absolute == 101  # ranked beyond fetched depth


def test_onpage_feature_extraction():
    page = PageContent(
        url="https://www.home-barista.com/descaling-guide", html=SAMPLE_HTML, ok=True, status=200
    )
    f = extract_onpage_features(page)
    assert f["has_schema"] == 1
    assert f["has_faq"] == 1  # FAQPage schema + <details>
    assert f["num_lists_tables"] == 2  # one <ul> + one <table>
    assert f["is_https"] == 1
    assert f["word_count"] > 10
    assert f["content_freshness_days"] >= 0  # parsed from meta


def test_semantic_and_entity_helpers():
    sim, passage = semantic_scores(
        "descale espresso machine", "This guide explains how to descale an espresso machine."
    )
    assert 0.0 <= sim <= 1.0 and 0.0 <= passage <= 1.0
    assert entity_overlap("descale espresso machine", "descaling an espresso machine") >= 1


def test_dry_run_produces_valid_schema():
    df, meta_df = build_dataset(
        ["how to descale an espresso machine"],
        crawl=False,
        fixture=str(FIXTURE),
        verbose=False,
    )
    # EXPECTED_COLUMNS must all be present, in order, as a prefix -- extra
    # reference columns (title, snippet, crawl_ok) are intentionally preserved
    # alongside them, not stripped (see finalise_dataset's docstring).
    assert list(df.columns[: len(EXPECTED_COLUMNS)]) == EXPECTED_COLUMNS
    assert set(EXPECTED_COLUMNS).issubset(df.columns)
    assert not df.isna().any().any(), "collected dataset has NaNs"
    assert df["cited"].sum() >= 1
    assert set(df["cited"].unique()).issubset({0, 1})
    # Query-level metadata: one row, with SERP-feature flags for clustering.
    assert len(meta_df) == 1
    assert meta_df.loc[0, "ai_overview_present"] == 1
    assert "has_local_pack" in meta_df.columns


def test_fetch_html_writes_cache_on_success(tmp_path):
    """fetch_html should save the raw HTML to cache_path when the fetch succeeds."""
    fake_resp = MagicMock(status_code=200, text="<html>hello</html>")
    fake_resp.headers = {"content-type": "text/html; charset=utf-8"}
    cache_path = tmp_path / "html" / "q0000__00.html"

    with patch("aio_gap_miner.collect.crawl.requests.get", return_value=fake_resp):
        page = fetch_html("https://example.com/page", cache_path=cache_path)

    assert page.ok is True
    assert cache_path.exists(), "cache file was not written"
    assert cache_path.read_text(encoding="utf-8") == "<html>hello</html>"


def test_fetch_html_no_cache_file_on_failure(tmp_path):
    """A failed fetch must not write a (misleading, empty) cache file."""
    fake_resp = MagicMock(status_code=404, text="")
    fake_resp.headers = {"content-type": "text/html"}
    cache_path = tmp_path / "html" / "q0000__00.html"

    with patch("aio_gap_miner.collect.crawl.requests.get", return_value=fake_resp):
        page = fetch_html("https://example.com/missing", cache_path=cache_path)

    assert page.ok is False
    assert not cache_path.exists(), "cache file must not exist for a failed fetch"


def test_collect_query_caches_raw_serp_json(tmp_path):
    """collect_query should write the exact raw DataForSEO response to disk
    when cache_dir is given, using the live-API code path (mocked client)."""
    raw_response = json.loads(FIXTURE.read_text())
    fake_client = MagicMock()
    fake_client.fetch_serp.return_value = raw_response

    rows, meta = collect_query(
        "how to descale an espresso machine",
        "q0000",
        fake_client,
        location_code=2276,
        language_code="de",
        max_organic=15,
        crawl=False,  # isolate JSON caching from HTML caching in this test
        fixture=None,  # forces the live-client code path where caching lives
        polite_delay=0,
        cache_dir=tmp_path,
    )

    cached = tmp_path / "serp_json" / "q0000.json"
    assert cached.exists(), "raw SERP JSON was not cached"
    assert json.loads(cached.read_text(encoding="utf-8")) == raw_response
    assert len(rows) > 0
    assert meta["ai_overview_present"] == 1


def test_collect_query_without_cache_dir_writes_nothing(tmp_path):
    """cache_dir=None (the --no-cache path) must not create any cache files."""
    raw_response = json.loads(FIXTURE.read_text())
    fake_client = MagicMock()
    fake_client.fetch_serp.return_value = raw_response

    collect_query(
        "how to descale an espresso machine",
        "q0000",
        fake_client,
        location_code=2276,
        language_code="de",
        max_organic=15,
        crawl=False,
        fixture=None,
        polite_delay=0,
        cache_dir=None,
    )

    assert list(tmp_path.iterdir()) == [], "no cache files should be written when cache_dir=None"


def test_feature_sets_variants_remove_leakage():
    """Variant A drops rank-101 rows; variant B keeps them but excludes rank
    from its feature list. Both exclude the leaky/dead columns."""
    import pandas as pd

    from aio_gap_miner.feature_sets import prepare_variant

    df = pd.DataFrame(
        {
            "query_id": ["q0", "q0", "q1", "q1"],
            "organic_rank": [1, 101, 5, 101],
            "cited": [0, 1, 0, 1],
            "word_count": [100, 200, 300, 400],
            "content_type": ["informational"] * 4,
        }
    )

    # Variant A: 101-rows removed, rank kept as a feature.
    df_a, num_a, cat_a = prepare_variant(df, "A")
    assert (df_a["organic_rank"] != 101).all()
    assert len(df_a) == 2
    assert "organic_rank" in num_a

    # Variant B: all rows kept, rank NOT a feature.
    df_b, num_b, cat_b = prepare_variant(df, "B")
    assert len(df_b) == 4
    assert "organic_rank" not in num_b
    assert "rank_reciprocal" not in num_b

    # Both drop the leaky/dead columns.
    for feats in (num_a, num_b):
        assert "domain_citation_rate" not in feats
        assert "domain_rating" not in feats
        assert "page_authority" not in feats
