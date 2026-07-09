"""Offline tests for the real-data collection pipeline.

No network required: SERP parsing runs against the bundled fixture, on-page
extraction runs against an inline HTML string, and the full assembly runs in
dry-run mode. These lock down the parts that don't need credentials -- the
label logic, feature extraction, and schema conformance.
"""

from __future__ import annotations

from pathlib import Path

from aio_gap_miner.collect.crawl import PageContent, extract_onpage_features
from aio_gap_miner.collect.pipeline import build_dataset, entity_overlap, semantic_scores
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
    df = build_dataset(
        ["how to descale an espresso machine"],
        crawl=False,
        fixture=str(FIXTURE),
        verbose=False,
    )
    assert list(df.columns) == EXPECTED_COLUMNS
    assert not df.isna().any().any(), "collected dataset has NaNs"
    assert df["cited"].sum() >= 1
    assert set(df["cited"].unique()).issubset({0, 1})
