#!/usr/bin/env python3
"""Extract deeper analytics from the cached raw data - no new API calls needed.

Your collection run already cached every raw DataForSEO JSON response and every
page's raw HTML under data/raw/_cache/. This script mines that cache for signals
we didn't extract the first time, and writes them as tidy CSVs you can chart.

Run from the project root:
    python scripts/extract_from_cache.py

Outputs (into reports/results/):
    serp_features_enriched.csv   - per query: every SERP feature + AIO stats + PAA count
    paa_questions.csv            - every "People Also Ask" question, one per row
    aio_text.csv                 - per query: the full AI Overview answer text + length
    aio_citations_detail.csv     - per cited URL: which AIO it came from, its snippet
    html_position.csv            - per page: where the first list/table/answer sits
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

CACHE = Path("data/raw/_cache")
SERP_DIR = CACHE / "serp_json"
HTML_DIR = CACHE / "html"
OUT = Path("reports/results")

FEATURE_TYPES = [
    "local_pack", "map", "people_also_ask", "knowledge_graph", "featured_snippet",
    "related_searches", "images", "video", "paid", "compare_sites",
    "people_also_search", "twitter", "top_stories", "shopping", "answer_box", "carousel",
]


def clean_domain(d) -> str:
    """Strip markdown-link wrapping and a leading www. from a domain string.

    DataForSEO sometimes returns reference domains wrapped as
    ``[www.example.com](https://www.example.com)``; normalise to ``example.com``.
    """
    if not isinstance(d, str):
        return ""
    m = re.match(r"\[([^\]]*)\]\([^)]*\)", d.strip())
    if m:
        d = m.group(1)
    d = d.strip().lower()
    return re.sub(r"^www\.", "", d)


def clean_url(u) -> str:
    """Unwrap a markdown-link-wrapped URL to the bare URL.

    Important beyond cosmetics: the wrapped form never matches the plain URLs in
    real.csv, so cleaning here is what lets downstream analyses find the cached
    HTML for a cited page at all.
    """
    if not isinstance(u, str):
        return ""
    m = re.match(r"\[[^\]]*\]\(([^)]*)\)", u.strip())
    if m:
        u = m.group(1)
    return u.strip()


def _iter_serp_files():
    for f in sorted(SERP_DIR.glob("*.json")):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            yield f.stem, raw["tasks"][0]["result"][0]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue


def _find_ai_overview(items: list[dict]) -> dict | None:
    for it in items:
        if it.get("type") == "ai_overview":
            return it
    return None


def _collect_aio_references(aio: dict) -> list[dict]:
    refs = list(aio.get("references") or [])
    for el in aio.get("items") or []:
        refs.extend(el.get("references") or [])
    return refs


def _aio_full_text(aio: dict) -> str:
    parts = []
    if aio.get("markdown"):
        parts.append(aio["markdown"])
    for el in aio.get("items") or []:
        if el.get("text"):
            parts.append(el["text"])
        if el.get("title"):
            parts.append(el["title"])
    return "\n".join(parts).strip()


def extract_serp_features():
    rows, paa_rows, aio_text_rows, cite_rows = [], [], [], []

    for qid, result in _iter_serp_files():
        query = result.get("keyword", "")
        items = result.get("items", [])
        item_types = result.get("item_types", [])
        present = {t: int(t in item_types) for t in FEATURE_TYPES}
        aio = _find_ai_overview(items)

        paa_questions = []
        for it in items:
            if it.get("type") == "people_also_ask":
                for el in it.get("items") or []:
                    q = el.get("title")
                    if q:
                        paa_questions.append(q)
                        paa_rows.append({"query_id": qid, "query": query, "paa_question": q})

        aio_present = aio is not None
        aio_text = _aio_full_text(aio) if aio else ""
        refs = _collect_aio_references(aio) if aio else []
        if aio_present:
            aio_text_rows.append({
                "query_id": qid, "query": query,
                "aio_char_len": len(aio_text),
                "aio_word_count": len(aio_text.split()),
                "aio_num_references": len(refs),
                "aio_text": aio_text[:5000],
            })
            for r in refs:
                cite_rows.append({
                    "query_id": qid, "query": query,
                    "cited_url": clean_url(r.get("url", "")),
                    "cited_domain": clean_domain(r.get("domain", "")),
                    "cited_title": r.get("title", ""),
                    "cited_snippet": (r.get("text", "") or "")[:500],
                })

        rows.append({
            "query_id": qid, "query": query,
            "se_results_count": result.get("se_results_count"),
            "items_count": result.get("items_count"),
            "num_paa_questions": len(paa_questions),
            "ai_overview_present": int(aio_present),
            "ai_overview_num_references": len(refs),
            "aio_word_count": len(aio_text.split()) if aio_present else 0,
            **{f"has_{t}": present[t] for t in FEATURE_TYPES},
        })

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "serp_features_enriched.csv", index=False)
    pd.DataFrame(paa_rows).to_csv(OUT / "paa_questions.csv", index=False)
    pd.DataFrame(aio_text_rows).to_csv(OUT / "aio_text.csv", index=False)
    pd.DataFrame(cite_rows).to_csv(OUT / "aio_citations_detail.csv", index=False)
    print(f"  serp_features_enriched.csv  ({len(rows)} queries)")
    print(f"  paa_questions.csv           ({len(paa_rows)} questions)")
    print(f"  aio_text.csv                ({len(aio_text_rows)} AI Overviews with text)")
    print(f"  aio_citations_detail.csv    ({len(cite_rows)} cited references)")


def extract_html_position():
    rows = []
    for hf in sorted(HTML_DIR.glob("*.html")):
        m = re.match(r"(q\d+)__(\d+)\.html", hf.name)
        if not m:
            continue
        qid, idx = m.group(1), int(m.group(2))
        try:
            html = hf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        total_len = max(len(soup.get_text(" ")), 1)

        def _pos_fraction(tag):
            if tag is None:
                return None
            chars_before = sum(len(s) for s in tag.find_all_previous(string=True))
            return round(chars_before / total_len, 3)

        first_p_words = None
        for p in soup.find_all("p"):
            w = len(p.get_text().split())
            if w >= 10:
                first_p_words = w
                break

        rows.append({
            "query_id": qid,
            "candidate_idx": idx,
            "first_list_pos": _pos_fraction(soup.find(["ul", "ol"])),
            "first_table_pos": _pos_fraction(soup.find("table")),
            "first_heading_pos": _pos_fraction(soup.find(["h2", "h3"])),
            "first_paragraph_words": first_p_words,
            "has_any_list": int(soup.find(["ul", "ol"]) is not None),
            "has_any_table": int(soup.find("table") is not None),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "html_position.csv", index=False)
    print(f"  html_position.csv           ({len(rows)} pages)")


def main():
    if not CACHE.exists():
        raise SystemExit(f"Cache not found at {CACHE}. Run from the project root.")
    print("Extracting SERP features, PAA questions, AIO text, citations ...")
    extract_serp_features()
    print("Extracting HTML structure/position features (this reads 6k+ files) ...")
    extract_html_position()
    print(f"\nDone. All outputs in {OUT}/")


if __name__ == "__main__":
    main()
