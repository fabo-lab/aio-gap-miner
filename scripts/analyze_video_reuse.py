#!/usr/bin/env python3
"""Do AI Overviews actually use what the video says - and point at the right moment?

Context
-------
76% of the AI Overviews here cite at least one YouTube video, and 97% of those
videos don't rank in the visible results. That makes "publish a video" an
attractive recommendation - but only if Google uses the content. If videos are
cited without their substance being used, the citation is decorative.

Two tests, in increasing strength:

  TEST 1 - REUSE. Does the AI Overview text share long word sequences with the
  video transcript? Same n-gram criterion used for web pages, with the same
  permutation null (transcripts paired against unrelated searches).

  TEST 2 - ALIGNMENT. 51% of the cited video URLs carry a `&t=` offset, so Google
  is pointing at a moment, not just a video. If the reused content sits near that
  moment in the transcript, Google demonstrably located it. If the matches are
  scattered randomly through the transcript, the timestamp means less than it
  appears to.

Test 2 is the one worth presenting: a positive result is direct evidence that the
video's spoken content was processed, not just its existence noted.

    python scripts/analyze_video_reuse.py
    python scripts/analyze_video_reuse.py --min-ngram 6
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")
CACHE = Path("data/raw/_cache/transcripts")
SEED = 42


def clean_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    text = re.sub(r"\[+\d*\]+", " ", text)
    text = re.sub(r"\(https?://[^)]*\)", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_words(text: str) -> list[str]:
    return re.findall(r"[a-zäöüßA-ZÄÖÜ0-9€%.,-]+", text.lower())


def ngrams(words: list[str], n: int) -> set[tuple]:
    if len(words) < n:
        return set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def split_sentences(text: str, min_words: int = 6) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.split()) >= min_words]


def transcript_windows(snippets: list[dict], window_s: float = 30.0) -> list[dict]:
    """Group timed snippets into overlapping windows so a match has a position."""
    if not snippets:
        return []
    out, cur, start = [], [], snippets[0].get("start", 0.0)
    for s in snippets:
        cur.append(s.get("text", ""))
        if s.get("start", 0.0) - start >= window_s:
            out.append({"start": start, "text": " ".join(cur)})
            cur, start = [], s.get("start", 0.0)
    if cur:
        out.append({"start": start, "text": " ".join(cur)})
    return out


def video_id_from_url(url: str) -> str | None:
    m = re.search(r"[?&]v=([A-Za-z0-9_-]+)", str(url))
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]+)", str(url))
    return m.group(1) if m else None


def timestamp_from_url(url: str) -> float | None:
    m = re.search(r"[?&]t=(\d+)", str(url))
    return float(m.group(1)) if m else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-ngram", type=int, default=6)
    p.add_argument("--permutations", type=int, default=5)
    p.add_argument("--window", type=float, default=30.0, help="Transcript window length (s).")
    args = p.parse_args()
    N = args.min_ngram

    aio_path = RESULTS / "aio_text.csv"
    if not aio_path.exists():
        raise SystemExit("Run scripts/extract_from_cache.py first.")
    if not CACHE.exists() or not list(CACHE.glob("*.json")):
        raise SystemExit(f"No transcripts in {CACHE}. Run fetch_youtube_transcripts.py first.")

    aio_df = pd.read_csv(aio_path)
    aio_sents = {r["query_id"]: split_sentences(clean_markdown(str(r["aio_text"])))
                 for _, r in aio_df.iterrows()}
    aio_sents = {k: v for k, v in aio_sents.items() if v}

    real = pd.read_csv("data/raw/real.csv")
    real["domain"] = real["url"].str.extract(r"https?://([^/]+)/", expand=False)
    yt = real[real["domain"].str.contains("youtu", na=False) & (real["cited"] == 1)].copy()
    yt["video_id"] = yt["url"].map(video_id_from_url)
    yt["cited_at_s"] = yt["url"].map(timestamp_from_url)
    yt = yt[yt["video_id"].notna()]

    transcripts = {}
    for f in CACHE.glob("*.json"):
        try:
            transcripts[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

    covered = yt[yt["video_id"].isin(transcripts)]
    print("=" * 76)
    print("  DO AI OVERVIEWS USE WHAT THE VIDEO SAYS?")
    print("=" * 76)
    print(f"\n  Cited video rows:        {len(yt)}")
    print(f"  Unique videos:           {yt['video_id'].nunique()}")
    print(f"  Transcripts available:   {len(transcripts)} "
          f"({covered['video_id'].nunique() / max(yt['video_id'].nunique(), 1):.0%} coverage)")
    auto = sum(1 for t in transcripts.values() if t.get("is_generated"))
    print(f"  ... auto-generated:      {auto} of {len(transcripts)}")
    print(f"  Citations with a &t= offset: {int(yt['cited_at_s'].notna().sum())} "
          f"({yt['cited_at_s'].notna().mean():.0%})")

    # ---------------------------------------------------------------- TEST 1
    print("\n" + "-" * 76)
    print(f"  TEST 1 - reuse: shared sequences of >= {N} consecutive words")
    print("-" * 76)

    pairs = covered.drop_duplicates(subset=["query_id", "video_id"])
    matches = []
    for _, row in pairs.iterrows():
        qid, vid = row["query_id"], row["video_id"]
        if qid not in aio_sents:
            continue
        tr = transcripts[vid]
        windows = transcript_windows(tr.get("snippets", []), args.window)
        if not windows:
            continue
        win_grams = [(w["start"], ngrams(normalise_words(w["text"]), N)) for w in windows]
        for sent in aio_sents[qid]:
            sg = ngrams(normalise_words(sent), N)
            if not sg:
                continue
            for start, wg in win_grams:
                if sg & wg:
                    matches.append({"query_id": qid, "query": row["query"], "video_id": vid,
                                    "aio_sentence": sent[:300], "match_at_s": start,
                                    "cited_at_s": row["cited_at_s"],
                                    "duration_s": tr.get("duration_s", 0)})
                    break

    obs = pd.DataFrame(matches)
    if len(obs):
        obs = obs.drop_duplicates(subset=["query_id", "video_id", "aio_sentence"])
    print(f"\n  Observed: {len(obs)} AI-Overview sentences match transcript content")
    if len(obs):
        print(f"    across {obs['query_id'].nunique()} searches and {obs['video_id'].nunique()} videos")

    # Permutation null: same videos, AI Overviews from unrelated searches.
    rng = random.Random(SEED)
    qids = list(aio_sents.keys())
    nulls = []
    for rep in range(args.permutations):
        cnt = 0
        for _, row in pairs.iterrows():
            vid = row["video_id"]
            fake = rng.choice([q for q in qids if q != row["query_id"]])
            tr = transcripts[vid]
            windows = transcript_windows(tr.get("snippets", []), args.window)
            win_grams = [ngrams(normalise_words(w["text"]), N) for w in windows]
            for sent in aio_sents[fake]:
                sg = ngrams(normalise_words(sent), N)
                if sg and any(sg & wg for wg in win_grams):
                    cnt += 1
        nulls.append(cnt)
        print(f"    null replicate {rep + 1}: {cnt}")
    null_mean = sum(nulls) / len(nulls) if nulls else 0
    if null_mean:
        print(f"\n  Observed {len(obs)} vs null {null_mean:.0f}  ->  "
              f"{len(obs) / null_mean:.1f}x above chance")
    else:
        print(f"\n  Observed {len(obs)} vs null 0 - no chance matches at this threshold")

    # ---------------------------------------------------------------- TEST 2
    print("\n" + "-" * 76)
    print("  TEST 2 - alignment: is the reused content near the cited moment?")
    print("-" * 76)
    aligned = obs[obs["cited_at_s"].notna()] if len(obs) else pd.DataFrame()
    if len(aligned) >= 5:
        aligned = aligned.copy()
        aligned["gap_s"] = (aligned["match_at_s"] - aligned["cited_at_s"]).abs()
        # A random position in the video is the comparison.
        rng2 = random.Random(SEED)
        aligned["random_gap_s"] = [
            abs(rng2.uniform(0, max(d, 1)) - c)
            for d, c in zip(aligned["duration_s"], aligned["cited_at_s"])
        ]
        print(f"\n  {len(aligned)} matches where Google supplied a timestamp")
        print(f"    median distance, match to cited moment: {aligned['gap_s'].median():.0f} s")
        print(f"    median distance if the match were random: {aligned['random_gap_s'].median():.0f} s")
        within = (aligned["gap_s"] <= 60).mean()
        within_rand = (aligned["random_gap_s"] <= 60).mean()
        print(f"    within 60 s of the cited moment: {within:.0%} "
              f"(random baseline {within_rand:.0%})")
        verdict = ("Google located the relevant passage" if within > within_rand + 0.15
                   else "no clear alignment - the timestamp may be generic")
        print(f"\n    -> {verdict}")

        FIGURES.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bins = range(0, int(max(aligned["gap_s"].max(), aligned["random_gap_s"].max())) + 30, 30)
        ax.hist(aligned["gap_s"], bins=bins, alpha=0.85, label="Actual match position",
                color="#4F46E5")
        ax.hist(aligned["random_gap_s"], bins=bins, alpha=0.5, label="Random position in video",
                color="#94A3B8")
        ax.set_xlabel("Distance from the moment Google linked to (seconds)")
        ax.set_ylabel("Number of matches")
        ax.set_title("Does Google point at the part of the video it actually used?")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "case_6_video_timestamp_alignment.png", dpi=130)
        plt.close(fig)
        print(f"    Chart -> {FIGURES}/case_6_video_timestamp_alignment.png")
    else:
        print(f"\n  Only {len(aligned)} timestamped matches - too few for the alignment test.")
        print("  Try a lower --min-ngram, or report Test 1 alone.")

    if len(obs):
        RESULTS.mkdir(parents=True, exist_ok=True)
        obs.to_csv(RESULTS / "video_transcript_reuse.csv", index=False)
        print(f"\n  Table -> {RESULTS}/video_transcript_reuse.csv")
        print("\n  Examples:")
        for _, r in obs.head(4).iterrows():
            at = f"{r['match_at_s']:.0f}s"
            cited = f", Google linked to {r['cited_at_s']:.0f}s" if pd.notna(r["cited_at_s"]) else ""
            print(f"\n    search: \"{r['query'][:50]}\"  (video {r['video_id']}, match at {at}{cited})")
            print(f"      AIO: {r['aio_sentence'][:150]}")


if __name__ == "__main__":
    main()
