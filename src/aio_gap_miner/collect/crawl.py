"""Crawl a candidate URL and extract the on-page features the model uses.

Everything here comes from the page's own HTML: how long it is, whether it uses
structured data (schema / FAQ), how much extractable structure it has (lists,
tables), how readable it is, how fresh it is, and a rough content-type guess.
These are the "content structure" signals that -- per the AI Overview research --
matter as much as ranking position for getting cited.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    import textstat

    _HAS_TEXTSTAT = True
except Exception:  # pragma: no cover
    _HAS_TEXTSTAT = False

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 aio-gap-miner/0.1"
)

_FORUM_DOMAINS = (
    "reddit.com",
    "quora.com",
    "stackexchange.com",
    "stackoverflow.com",
    "gutefrage.net",
)
_VIDEO_DOMAINS = ("youtube.com", "youtu.be", "vimeo.com")


@dataclass
class PageContent:
    url: str
    html: str = ""
    ok: bool = False
    status: int = 0


def fetch_html(url: str, timeout: int = 20, cache_path: Path | None = None) -> PageContent:
    """Fetch a URL's HTML. Never raises -- returns ``ok=False`` on any failure.

    If ``cache_path`` is given, the raw HTML is also written there on success --
    a permanent local copy so future feature-extraction changes (or entirely
    different analyses) never require re-crawling the same page.
    """
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _UA},
            timeout=timeout,
            allow_redirects=True,
        )
        ok = resp.status_code == 200 and "text/html" in resp.headers.get("content-type", "")
        if ok and cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(resp.text, encoding="utf-8", errors="replace")
            except OSError:
                pass  # caching is a bonus, never let it break the actual crawl
        return PageContent(url=url, html=resp.text if ok else "", ok=ok, status=resp.status_code)
    except Exception:
        return PageContent(url=url, ok=False, status=0)


def _iter_jsonld(soup: BeautifulSoup) -> list[dict]:
    """Return all JSON-LD blocks as dicts (flattening @graph)."""
    blocks: list[dict] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue
        if isinstance(data, dict):
            blocks.extend(data.get("@graph", [data]))
        elif isinstance(data, list):
            blocks.extend(x for x in data if isinstance(x, dict))
    return blocks


def _schema_types(blocks: list[dict]) -> set[str]:
    types: set[str] = set()
    for b in blocks:
        t = b.get("@type")
        if isinstance(t, list):
            types.update(str(x) for x in t)
        elif t:
            types.add(str(t))
    return types


def _parse_freshness_days(soup: BeautifulSoup, blocks: list[dict]) -> float | None:
    """Days since the page was last modified, from meta/JSON-LD/time tags."""
    candidates: list[str] = []
    for b in blocks:
        for key in ("dateModified", "datePublished"):
            if b.get(key):
                candidates.append(str(b[key]))
    for prop in ("article:modified_time", "article:published_time"):
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            candidates.append(tag["content"])
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        candidates.append(time_tag["datetime"])

    for raw in candidates:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            days = (datetime.now(timezone.utc) - dt).days
            if days >= 0:
                return float(days)
        except Exception:
            continue
    return None


def _guess_content_type(url: str, title: str, schema_types: set[str]) -> str:
    """Rough informational / commercial / transactional label."""
    u = (url + " " + title).lower()
    if "Product" in schema_types or any(
        k in u for k in ("/product", "/buy", "/pricing", "/shop", "kaufen", "preis")
    ):
        return "transactional"
    if any(k in u for k in ("best", "review", " vs ", "vergleich", "test", "beste")):
        return "commercial"
    return "informational"


def extract_onpage_features(page: PageContent, title_hint: str = "") -> dict:
    """Parse HTML into the on-page portion of the feature schema.

    On failure (page not fetched) returns NaN/neutral values so the row can
    still be assembled; the pipeline imputes them afterwards.
    """
    import numpy as np

    domain = urlparse(page.url).netloc.lower()
    is_https = int(page.url.lower().startswith("https"))
    is_forum = int(any(d in domain for d in _FORUM_DOMAINS))
    is_video_domain = any(d in domain for d in _VIDEO_DOMAINS)

    if not page.ok or not page.html:
        return {
            "word_count": np.nan,
            "has_schema": np.nan,
            "has_faq": np.nan,
            "num_lists_tables": np.nan,
            "readability_score": np.nan,
            "content_freshness_days": np.nan,
            "is_https": is_https,
            "is_forum": is_forum,
            "is_video": int(is_video_domain),
            "content_type": "informational",
            "page_text": "",
        }

    soup = BeautifulSoup(page.html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    word_count = len(text.split())

    blocks = _iter_jsonld(BeautifulSoup(page.html, "lxml"))
    schema_types = _schema_types(blocks)
    has_schema = int(bool(blocks) or bool(soup.find(attrs={"itemtype": True})))

    faq_text = "frequently asked" in text.lower() or "häufige fragen" in text.lower()
    has_faq = int("FAQPage" in schema_types or bool(soup.find("details")) or faq_text)

    num_lists_tables = len(soup.find_all(["ul", "ol", "table"]))

    if _HAS_TEXTSTAT and word_count > 30:
        try:
            readability = float(textstat.flesch_reading_ease(text[:20000]))
            readability = max(0.0, min(100.0, readability))
        except Exception:
            readability = np.nan
    else:
        readability = np.nan

    freshness = _parse_freshness_days(soup, blocks)
    title = title_hint or (soup.title.string if soup.title else "") or ""
    content_type = _guess_content_type(page.url, title, schema_types)

    has_video_schema = "VideoObject" in schema_types or bool(soup.find("video"))
    is_video = int(is_video_domain or has_video_schema)

    return {
        "word_count": int(word_count),
        "has_schema": has_schema,
        "has_faq": has_faq,
        "num_lists_tables": int(min(num_lists_tables, 50)),
        "readability_score": readability,
        "content_freshness_days": freshness if freshness is not None else np.nan,
        "is_https": is_https,
        "is_forum": is_forum,
        "is_video": is_video,
        "content_type": content_type,
        "page_text": text,
    }
