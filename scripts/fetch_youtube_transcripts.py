#!/usr/bin/env python3
"""Fetch transcripts for every YouTube video an AI Overview cited.

Why
---
76% of the AI Overviews in this dataset cite at least one YouTube video, and 97%
of those videos don't rank in the visible results. That makes "publish a video" a
tempting recommendation - but it only holds if Google actually *uses* what the
video says. If videos are cited without their content being used, the citation is
decorative and the recommendation is much weaker.

Answering that needs the transcripts. This script collects them.

Transcripts are cached under data/raw/_cache/transcripts/ exactly like the HTML
and SERP caches, so this only has to run once and can be resumed if interrupted.

    python scripts/fetch_youtube_transcripts.py
    python scripts/fetch_youtube_transcripts.py --limit 20   # try a few first

Notes
-----
* Not every video has a transcript. Auto-generated captions count and are used;
  the script records which is which.
* YouTube rate-limits. The default delay is deliberately polite. If you start
  seeing IpBlocked errors, stop, wait, and resume later - the cache means nothing
  is lost.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

CACHE = Path("data/raw/_cache/transcripts")
RESULTS = Path("reports/results")


def video_id_from_url(url: str) -> str | None:
    """Extract the YouTube video id from any of its URL shapes."""
    if not isinstance(url, str):
        return None
    try:
        p = urlparse(url)
    except ValueError:
        return None
    host = (p.netloc or "").lower().replace("www.", "")
    if "youtube.com" in host:
        if p.path == "/watch":
            vid = parse_qs(p.query).get("v", [None])[0]
            return vid
        m = re.match(r"^/(?:embed|shorts|v)/([A-Za-z0-9_-]{6,})", p.path)
        if m:
            return m.group(1)
    if "youtu.be" in host:
        seg = p.path.lstrip("/").split("/")[0]
        return seg or None
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/raw/real.csv")
    p.add_argument("--limit", type=int, default=None, help="Only fetch the first N videos.")
    p.add_argument("--delay", type=float, default=1.5, help="Seconds between requests.")
    p.add_argument("--languages", default="de,en", help="Preferred transcript languages, in order.")
    args = p.parse_args()

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import CouldNotRetrieveTranscript
    except ImportError:
        raise SystemExit("pip install youtube-transcript-api")

    df = pd.read_csv(args.data)
    df["domain"] = df["url"].str.extract(r"https?://([^/]+)/", expand=False)
    yt = df[df["domain"].str.contains("youtu", na=False) & (df["cited"] == 1)].copy()
    yt["video_id"] = yt["url"].map(video_id_from_url)
    yt = yt[yt["video_id"].notna()]

    unique = yt.drop_duplicates(subset=["video_id"])
    print(f"Cited YouTube rows: {len(yt)} | unique videos: {len(unique)}")

    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [v for v in unique["video_id"] if not (CACHE / f"{v}.json").exists()]
    already = len(unique) - len(todo)
    if already:
        print(f"Already cached: {already}")
    if args.limit:
        todo = todo[: args.limit]
    print(f"Fetching: {len(todo)}\n")

    api = YouTubeTranscriptApi()
    langs = [s.strip() for s in args.languages.split(",") if s.strip()]
    ok, failed = 0, {}

    for i, vid in enumerate(todo, 1):
        try:
            fetched = api.fetch(vid, languages=langs)
            snippets = fetched.to_raw_data()
            text = " ".join(s.get("text", "") for s in snippets)
            payload = {
                "video_id": vid,
                "language": getattr(fetched, "language", None),
                "language_code": getattr(fetched, "language_code", None),
                "is_generated": getattr(fetched, "is_generated", None),
                "n_snippets": len(snippets),
                "duration_s": round(snippets[-1].get("start", 0) + snippets[-1].get("duration", 0), 1)
                if snippets else 0,
                "text": re.sub(r"\s+", " ", text).strip(),
                # Timed snippets are kept because Google cites a *moment* in the
                # video (51% of citations carry a &t= offset). Keeping the timing
                # is what allows checking whether the cited moment is actually
                # where the reused content sits.
                "snippets": [
                    {"start": round(s.get("start", 0), 1), "text": s.get("text", "")}
                    for s in snippets
                ],
            }
            (CACHE / f"{vid}.json").write_text(json.dumps(payload, ensure_ascii=False),
                                               encoding="utf-8")
            ok += 1
            print(f"[{i}/{len(todo)}] {vid}  ok  "
                  f"({payload['language_code']}, "
                  f"{'auto' if payload['is_generated'] else 'manual'}, "
                  f"{len(payload['text'].split())} words)", flush=True)
        except CouldNotRetrieveTranscript as exc:
            reason = type(exc).__name__
            failed[reason] = failed.get(reason, 0) + 1
            print(f"[{i}/{len(todo)}] {vid}  skipped ({reason})", flush=True)
        except Exception as exc:  # noqa: BLE001 - never let one video stop the run
            reason = type(exc).__name__
            failed[reason] = failed.get(reason, 0) + 1
            print(f"[{i}/{len(todo)}] {vid}  error ({reason})", flush=True)
        if args.delay:
            time.sleep(args.delay)

    total_cached = len(list(CACHE.glob("*.json")))
    print(f"\nFetched {ok} | cached in total {total_cached} of {len(unique)} videos")
    if failed:
        print("Not retrievable:")
        for reason, n in sorted(failed.items(), key=lambda kv: -kv[1]):
            print(f"  {n:4d}  {reason}")
        print("\n  A missing transcript is not a failure of the analysis - it just means")
        print("  that video can't be checked. Report the coverage rate.")

    # Map video -> query so the analysis can pair them up.
    RESULTS.mkdir(parents=True, exist_ok=True)
    yt[["query_id", "query", "url", "video_id", "title"]].drop_duplicates().to_csv(
        RESULTS / "cited_videos.csv", index=False)
    print(f"\nVideo/search mapping -> {RESULTS}/cited_videos.csv")
    print(f"Transcripts           -> {CACHE}/")


if __name__ == "__main__":
    main()
