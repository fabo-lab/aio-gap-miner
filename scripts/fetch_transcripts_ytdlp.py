#!/usr/bin/env python3
"""Fetch the remaining transcripts via yt-dlp, when the API route is IP-blocked.

`youtube-transcript-api` and `yt-dlp` reach YouTube differently, so a block on
one often doesn't apply to the other. This script covers whatever the API route
couldn't get, converts the subtitle files into the same JSON shape the analysis
already reads, and skips anything already cached.

    python scripts/fetch_transcripts_ytdlp.py            # everything still missing
    python scripts/fetch_transcripts_ytdlp.py --limit 5  # try a few first

Requires `pip install yt-dlp`.

Note on auto-generated captions: YouTube's VTT for those uses a rolling display
where each cue repeats part of the previous one. The parser below removes that
duplication, otherwise the transcript text would be inflated several times over
and every overlap measurement with it would be wrong.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

CACHE = Path("data/raw/_cache/transcripts")
TMP = Path("data/raw/_cache/_vtt_tmp")


def video_id_from_url(url: str) -> str | None:
    m = re.search(r"[?&]v=([A-Za-z0-9_-]+)", str(url))
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]+)", str(url))
    return m.group(1) if m else None


def _ts_to_seconds(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s.replace(",", "."))


def parse_vtt(path: Path) -> list[dict]:
    """VTT -> [{start, text}], with the rolling-caption duplication removed."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    snippets: list[dict] = []
    seen: set[str] = set()
    cur_start: float | None = None
    buf: list[str] = []

    def flush():
        nonlocal buf, cur_start
        if cur_start is None or not buf:
            buf = []
            return
        text = " ".join(buf).strip()
        text = re.sub(r"<[^>]+>", "", text)          # inline timing tags
        text = re.sub(r"\s+", " ", text).strip()
        if text and text not in seen:
            seen.add(text)
            snippets.append({"start": round(cur_start, 1), "text": text})
        buf = []

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        m = re.match(r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})", line)
        if m:
            flush()
            cur_start = _ts_to_seconds(m.group(1))
            continue
        if line.isdigit():
            continue
        buf.append(line)
    flush()
    return snippets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/raw/real.csv")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--delay", type=float, default=2.0)
    p.add_argument("--lang", default="de")
    args = p.parse_args()

    if subprocess.run(["which", "yt-dlp"], capture_output=True).returncode != 0:
        raise SystemExit("yt-dlp not found. Run: pip install yt-dlp")

    df = pd.read_csv(args.data)
    df["domain"] = df["url"].str.extract(r"https?://([^/]+)/", expand=False)
    yt = df[df["domain"].str.contains("youtu", na=False) & (df["cited"] == 1)].copy()
    yt["video_id"] = yt["url"].map(video_id_from_url)
    ids = sorted(set(yt["video_id"].dropna()))

    CACHE.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    todo = [v for v in ids if not (CACHE / f"{v}.json").exists()]
    print(f"Unique cited videos: {len(ids)} | already cached: {len(ids) - len(todo)} "
          f"| to fetch: {len(todo)}")
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("Nothing to do.")
        return
    print()

    ok, failed = 0, {}
    for i, vid in enumerate(todo, 1):
        url = f"https://www.youtube.com/watch?v={vid}"
        cmd = [
            "yt-dlp", "--skip-download",
            "--write-auto-sub", "--write-sub",
            "--sub-lang", args.lang, "--sub-format", "vtt",
            "--no-warnings", "--quiet",
            "-o", str(TMP / "%(id)s"), url,
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            failed["timeout"] = failed.get("timeout", 0) + 1
            print(f"[{i}/{len(todo)}] {vid}  timeout", flush=True)
            continue

        vtts = sorted(TMP.glob(f"{vid}*.vtt"))
        if not vtts:
            reason = "no subtitles"
            if res.stderr and "Sign in" in res.stderr:
                reason = "login required"
            elif res.returncode != 0:
                reason = "yt-dlp error"
            failed[reason] = failed.get(reason, 0) + 1
            print(f"[{i}/{len(todo)}] {vid}  skipped ({reason})", flush=True)
            continue

        snippets = parse_vtt(vtts[0])
        if not snippets:
            failed["empty"] = failed.get("empty", 0) + 1
            print(f"[{i}/{len(todo)}] {vid}  skipped (empty transcript)", flush=True)
        else:
            text = re.sub(r"\s+", " ", " ".join(s["text"] for s in snippets)).strip()
            payload = {
                "video_id": vid,
                "language": args.lang,
                "language_code": args.lang,
                "is_generated": "auto" in vtts[0].name or True,
                "n_snippets": len(snippets),
                "duration_s": snippets[-1]["start"],
                "text": text,
                "snippets": snippets,
                "source": "yt-dlp",
            }
            (CACHE / f"{vid}.json").write_text(json.dumps(payload, ensure_ascii=False),
                                               encoding="utf-8")
            ok += 1
            print(f"[{i}/{len(todo)}] {vid}  ok ({len(text.split())} words, "
                  f"{len(snippets)} cues)", flush=True)
        for f in vtts:
            f.unlink(missing_ok=True)
        if args.delay:
            time.sleep(args.delay)

    total = len(list(CACHE.glob("*.json")))
    print(f"\nFetched {ok} | cached in total {total} of {len(ids)} videos "
          f"({total / len(ids):.0%} coverage)")
    if failed:
        print("Not retrievable:")
        for reason, n in sorted(failed.items(), key=lambda kv: -kv[1]):
            print(f"  {n:4d}  {reason}")
    try:
        TMP.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
