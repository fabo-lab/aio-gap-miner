"""DataForSEO SERP + AI Overview client and parser.

Fetches Google SERPs with the AI Overview feature and extracts, per query:
the organic candidates (URL + rank) and the set of URLs Google *cited* in its
AI Overview. Those two together give the training signal: a (query, URL) pair
is labelled ``cited = 1`` if the URL appears in the AI Overview references.

Endpoint (verified against DataForSEO docs, 2026):
``POST https://api.dataforseo.com/v3/serp/google/organic/live/advanced``
with ``load_async_ai_overview: true`` so asynchronous AI Overviews are fetched
too. Cited sources live in the ``ai_overview`` item's ``references`` array
(objects of type ``ai_overview_reference`` with a ``url`` field), both at the
top level and inside each ``ai_overview_element``.

Auth is HTTP Basic with your DataForSEO login/password, read from the
environment: ``DATAFORSEO_LOGIN`` and ``DATAFORSEO_PASSWORD``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests

DATAFORSEO_ENDPOINT = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"

# Germany by default (Munich market). US = 2840.
DEFAULT_LOCATION_CODE = 2276
DEFAULT_LANGUAGE_CODE = "de"


def normalize_url(url: str) -> str:
    """Canonicalise a URL for matching (lowercase host, strip trailing slash/fragment)."""
    try:
        p = urlparse(url.strip())
        host = (p.netloc or "").lower()
        path = p.path.rstrip("/") or "/"
        return f"{p.scheme.lower()}://{host}{path}"
    except Exception:
        return url.strip()


@dataclass
class Candidate:
    """One candidate URL for a query (a potential AI Overview source)."""

    url: str
    domain: str
    rank_absolute: int  # 101 sentinel if cited but not in the organic block
    title: str = ""
    snippet: str = ""
    cited: bool = False


@dataclass
class SerpResult:
    """Parsed SERP for one query."""

    query: str
    candidates: list[Candidate] = field(default_factory=list)
    aio_present: bool = False

    @property
    def cited_urls(self) -> set[str]:
        return {normalize_url(c.url) for c in self.candidates if c.cited}


class DataForSEOClient:
    """Thin client for the DataForSEO SERP Advanced endpoint."""

    def __init__(self, login: str | None = None, password: str | None = None):
        self.login = login or os.environ.get("DATAFORSEO_LOGIN")
        self.password = password or os.environ.get("DATAFORSEO_PASSWORD")
        if not self.login or not self.password:
            raise RuntimeError(
                "DataForSEO credentials missing. Set DATAFORSEO_LOGIN and "
                "DATAFORSEO_PASSWORD environment variables (get them from "
                "https://app.dataforseo.com/api-access)."
            )

    def fetch_serp(
        self,
        keyword: str,
        location_code: int = DEFAULT_LOCATION_CODE,
        language_code: str = DEFAULT_LANGUAGE_CODE,
        timeout: int = 60,
    ) -> dict:
        """POST one keyword to the live SERP Advanced endpoint; return raw JSON."""
        payload = [
            {
                "keyword": keyword,
                "location_code": location_code,
                "language_code": language_code,
                "load_async_ai_overview": True,
                "expand_ai_overview": True,
            }
        ]
        resp = requests.post(
            DATAFORSEO_ENDPOINT,
            auth=(self.login, self.password),
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()


def _collect_references(items: list[dict]) -> list[dict]:
    """Pull every ai_overview_reference from an ai_overview item (top level + nested)."""
    refs: list[dict] = []
    for item in items:
        if item.get("type") != "ai_overview":
            continue
        refs.extend(item.get("references") or [])
        for element in item.get("items") or []:
            refs.extend(element.get("references") or [])
    return refs


def parse_serp(raw: dict, query: str) -> SerpResult:
    """Turn a raw DataForSEO response into a :class:`SerpResult`.

    Robust to the usual missing/empty keys. Candidate set = union of organic
    results and AI Overview references; ``cited`` is set for URLs that appear in
    the references. A cited URL not present in the organic block gets a rank
    sentinel of 101 (it's ranked beyond the fetched depth, which is common --
    ~40% of cited pages rank 11-20).
    """
    result = SerpResult(query=query)

    tasks = raw.get("tasks") or []
    if not tasks:
        return result
    results = tasks[0].get("result") or []
    if not results:
        return result
    items = results[0].get("items") or []

    # 1) Cited URLs from the AI Overview references.
    references = _collect_references(items)
    cited_by_url: dict[str, dict] = {}
    for ref in references:
        url = ref.get("url")
        if not url:
            continue
        cited_by_url[normalize_url(url)] = ref
    result.aio_present = any(it.get("type") == "ai_overview" for it in items)

    # 2) Organic candidates.
    seen: set[str] = set()
    for item in items:
        if item.get("type") != "organic":
            continue
        url = item.get("url")
        if not url:
            continue
        key = normalize_url(url)
        seen.add(key)
        result.candidates.append(
            Candidate(
                url=url,
                domain=item.get("domain", urlparse(url).netloc),
                rank_absolute=int(item.get("rank_absolute", 101)),
                title=item.get("title", "") or "",
                snippet=item.get("description", "") or "",
                cited=key in cited_by_url,
            )
        )

    # 3) Cited URLs not present in the organic block become extra candidates.
    for key, ref in cited_by_url.items():
        if key in seen:
            continue
        url = ref.get("url")
        result.candidates.append(
            Candidate(
                url=url,
                domain=ref.get("domain", urlparse(url).netloc),
                rank_absolute=101,  # ranked beyond fetched depth
                title=ref.get("title", "") or "",
                snippet=ref.get("text", "") or "",
                cited=True,
            )
        )

    return result


def load_serp_fixture(path: str | Path, query: str) -> SerpResult:
    """Parse a saved DataForSEO JSON fixture (used for --dry-run and tests)."""
    raw = json.loads(Path(path).read_text())
    return parse_serp(raw, query)
