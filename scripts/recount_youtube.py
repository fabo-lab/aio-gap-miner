#!/usr/bin/env python3
"""Re-extract the YouTube citation facts correctly, straight from the raw cache.

The bug this works around
-------------------------
`collect/serp.py::normalize_url()` drops the query string when building the key
for the cited-URL dictionary:

    return f"{p.scheme.lower()}://{host}{path}"    # ?v=... discarded

Every `youtube.com/watch?v=...` URL therefore collapses to the same key
`https://www.youtube.com/watch`, and because the references go into a dict, only
the last one per search survives. The evidence is unambiguous: of 249 searches
with a `watch?v=` citation, exactly **zero** have more than one - while Shorts,
which carry the id in the path and so don't collide, show 3 of 62 with several.

So every per-video number derived from `real.csv` is a lower bound, not a count.

This script goes back to the cached DataForSEO JSON and counts the AI Overview
references directly, with no normalisation and no deduplication. Those files are
the untouched API responses, so the counts here are the real ones.

What it does NOT fix: the labels in `real.csv`, which were written during
collection with the same buggy matcher. That affects an enumerable 11 rows
(pages appearing twice in one search, once ranked with `cited=0` and once as a
rank-101 row with `cited=1`, because http/https didn't match). 11 of 4,857 moves
no result, and rebuilding the training data would invalidate every analysis
downstream - so it is documented rather than silently patched.

    python scripts/recount_youtube.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

SERP_DIR = Path("data/raw/_cache/serp_json")
RESULTS = Path("reports/results")


def video_key(url: str) -> str | None:
    """A stable identifier for a YouTube URL - watch id, short id, or youtu.be id."""
    if not isinstance(url, str) or "youtu" not in url.lower():
        return None
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    m = re.search(r"/(?:shorts|embed|v)/([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,})", url)
    return m.group(1) if m else None


def timestamp(url: str) -> int | None:
    m = re.search(r"[?&]t=(\d+)", str(url))
    return int(m.group(1)) if m else None


def collect_refs(items: list[dict]) -> list[dict]:
    refs = []
    for it in items:
        if it.get("type") != "ai_overview":
            continue
        refs.extend(it.get("references") or [])
        for el in it.get("items") or []:
            refs.extend(el.get("references") or [])
    return refs


def main() -> None:
    if not SERP_DIR.exists():
        raise SystemExit(f"Cache not found at {SERP_DIR}")

    rows = []
    n_aio = 0
    for f in sorted(SERP_DIR.glob("*.json")):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            result = raw["tasks"][0]["result"][0]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue
        items = result.get("items", [])
        if not any(it.get("type") == "ai_overview" for it in items):
            continue
        n_aio += 1
        query = result.get("keyword", "")
        for ref in collect_refs(items):
            url = ref.get("url", "")
            vid = video_key(url)
            if vid:
                rows.append({
                    "query_id": f.stem, "query": query, "video_id": vid, "url": url,
                    "is_short": "/shorts/" in url, "t": timestamp(url),
                    "title": ref.get("title", ""),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No YouTube references found in the cache.")
    # Exact duplicates within one search are the same reference listed twice.
    df = df.drop_duplicates(subset=["query_id", "video_id"])

    print("=" * 74)
    print("  YOUTUBE CITATIONS - recounted from the raw cache")
    print("=" * 74)
    print(f"\n  AI Overviews in the cache:            {n_aio}")
    print(f"  YouTube citations (corrected):        {len(df)}")
    print(f"  Distinct videos (corrected):          {df['video_id'].nunique()}")
    q_with = df["query_id"].nunique()
    print(f"  AI Overviews citing >=1 video:        {q_with} ({q_with / n_aio:.0%})")

    per_q = df.groupby("query_id").size()
    print(f"\n  Searches citing more than one video:  {int((per_q > 1).sum())} "
          f"({(per_q > 1).mean():.0%})  <- was 0 before the fix for watch URLs")
    print(f"  Max videos cited in one search:       {int(per_q.max())}")
    print(f"  Mean videos per citing search:        {per_q.mean():.2f}")

    print(f"\n  Shorts among citations:               {int(df['is_short'].sum())} "
          f"({df['is_short'].mean():.0%})")
    with_t = df["t"].notna()
    print(f"  Citations carrying a &t= offset:      {int(with_t.sum())} "
          f"({with_t.mean():.0%} of all citations)")
    non_short = df[~df["is_short"]]
    if len(non_short):
        print(f"    ... among non-Shorts only:          "
              f"{non_short['t'].notna().mean():.0%}  (Shorts cannot carry one)")

    # Concentration
    counts = df["video_id"].value_counts()
    print(f"\n  Concentration:")
    print(f"    top 1 video:  {counts.iloc[0]:4d} citations "
          f"({counts.iloc[0] / len(df):.0%})")
    for k in (5, 10):
        if len(counts) >= k:
            print(f"    top {k:2d} videos: {counts.head(k).sum():4d} citations "
                  f"({counts.head(k).sum() / len(df):.0%})")

    # Is the timestamp a property of the answer, or of the video?
    print("\n  Is the timestamp specific to the answer, or fixed per video?")
    ts_by_video = defaultdict(set)
    for _, r in df[with_t].iterrows():
        ts_by_video[r["video_id"]].add(int(r["t"]))
    multi = {v: t for v, t in ts_by_video.items()
             if (df[with_t]["video_id"] == v).sum() > 1}
    constant = sum(1 for t in multi.values() if len(t) == 1)
    if multi:
        print(f"    Videos cited more than once with a timestamp: {len(multi)}")
        print(f"    ... always the SAME timestamp:                {constant} "
              f"({constant / len(multi):.0%})")
        print("\n    Where the timestamp never varies, it is a property of the video")
        print("    (a chapter or most-replayed marker), not of the answer. So it")
        print("    cannot be read as 'Google located the relevant moment'.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "youtube_citations_corrected.csv", index=False)
    summary = pd.DataFrame([{
        "ai_overviews": n_aio,
        "youtube_citations": len(df),
        "distinct_videos": int(df["video_id"].nunique()),
        "aio_citing_video": q_with,
        "share_aio_citing_video": round(q_with / n_aio, 3),
        "searches_citing_multiple": int((per_q > 1).sum()),
        "share_with_timestamp": round(float(with_t.mean()), 3),
        "share_shorts": round(float(df["is_short"].mean()), 3),
        "top1_share": round(float(counts.iloc[0] / len(df)), 3),
        "top5_share": round(float(counts.head(5).sum() / len(df)), 3),
        "videos_with_constant_timestamp": constant,
        "videos_multi_cited_with_timestamp": len(multi),
    }])
    summary.to_csv(RESULTS / "youtube_summary_corrected.csv", index=False)
    print(f"\n  Tables -> {RESULTS}/youtube_citations_corrected.csv, youtube_summary_corrected.csv")

    print("\n" + "=" * 74)
    print("  Which numbers change, and which survive")
    print("=" * 74)
    print("  SURVIVES: the share of AI Overviews citing at least one video. A")
    print("            'at least one' statement is unaffected by which video won")
    print("            the dictionary slot - one always remained.")
    print("  CHANGES:  total citations, distinct videos, the most-cited video, the")
    print("            timestamp share, and every concentration figure.")
    print("  DROP:     'Google links a moment, not a video' - the timestamp is")
    print("            constant per video in most repeat cases.")


if __name__ == "__main__":
    main()
