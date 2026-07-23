#!/usr/bin/env python3
"""Does Google link to the part of the video it actually used? (fuzzy version)

Why a second version
--------------------
The first attempt (`analyze_video_reuse.py`) looked for exact shared word
sequences. On this data that fails almost by construction: 28 of 30 available
transcripts are auto-generated, so they carry no punctuation, no capitalisation
and regular transcription errors. A single misheard word breaks a 6-word
sequence. It found 5 matches against a null of ~2 - a non-result, and one that
says more about the method than about Google.

This version asks the same question with a measure that survives noisy text.

The question: **65% of cited video URLs carry a `&t=` offset, so Google points at
a moment. Is the content the AI Overview used actually located near that moment?**

Method: split the transcript into overlapping time windows, score each window by
word overlap with the AI Overview text, take the best-scoring window, and measure
how far it sits from the moment Google linked to. Compare against two nulls:

  * a random position in the same video (is the alignment better than chance?)
  * the same procedure with an AI Overview from an unrelated search (is the match
    driven by the actual content, or by generic vocabulary?)

A clear result in either direction is informative. If the best-matching window
sits near the cited moment, Google demonstrably located the passage. If not, the
timestamp carries less meaning than it appears to.

    python scripts/analyze_video_alignment.py
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
import numpy as np
import pandas as pd

RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")
CACHE = Path("data/raw/_cache/transcripts")
SEED = 42

# Very common German words carry no signal about *which* passage matches.
STOP = {
    "der", "die", "das", "und", "ist", "ich", "für", "wie", "was", "wo", "ein",
    "eine", "einen", "einem", "einer", "man", "kann", "sich", "auf", "mit", "von",
    "zu", "im", "in", "den", "dem", "des", "es", "am", "an", "bei", "wird",
    "werden", "sind", "hat", "haben", "oder", "auch", "als", "nach", "aus", "um",
    "so", "dann", "aber", "wenn", "dass", "sie", "wir", "er", "nicht", "noch",
    "nur", "sehr", "schon", "mehr", "bis", "vor", "über", "unter", "durch", "zum",
    "zur", "diese", "dieser", "dieses", "da", "hier", "also", "ja", "mal",
}


def clean_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    text = re.sub(r"\[+\d*\]+", " ", text)
    text = re.sub(r"\(https?://[^)]*\)", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-zäöüßA-ZÄÖÜ0-9]{3,}", str(text).lower())
    return {w for w in words if w not in STOP}


def windows(snippets: list[dict], length_s: float, step_s: float) -> list[dict]:
    """Overlapping time windows, so a match isn't split across a boundary."""
    if not snippets:
        return []
    end = snippets[-1].get("start", 0.0)
    out = []
    t = 0.0
    while t <= end:
        text = " ".join(s.get("text", "") for s in snippets
                        if t <= s.get("start", 0.0) < t + length_s)
        if text.strip():
            out.append({"start": t, "words": content_words(text)})
        t += step_s
    return out


def best_window(aio_words: set[str], wins: list[dict]) -> tuple[float, float]:
    """Return (start of best window, overlap score) using Jaccard-style overlap."""
    best_s, best_score = 0.0, 0.0
    for w in wins:
        if not w["words"]:
            continue
        inter = len(aio_words & w["words"])
        if inter == 0:
            continue
        score = inter / (len(w["words"]) ** 0.5)  # length-normalised
        if score > best_score:
            best_score, best_s = score, w["start"]
    return best_s, best_score


def video_id_from_url(url: str) -> str | None:
    m = re.search(r"[?&]v=([A-Za-z0-9_-]+)", str(url))
    return m.group(1) if m else None


def timestamp_from_url(url: str) -> float | None:
    m = re.search(r"[?&]t=(\d+)", str(url))
    return float(m.group(1)) if m else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", type=float, default=45.0)
    p.add_argument("--step", type=float, default=15.0)
    args = p.parse_args()

    if not CACHE.exists() or not list(CACHE.glob("*.json")):
        raise SystemExit(f"No transcripts in {CACHE}. Run fetch_youtube_transcripts.py first.")
    aio_path = RESULTS / "aio_text.csv"
    if not aio_path.exists():
        raise SystemExit("Run scripts/extract_from_cache.py first.")

    aio_df = pd.read_csv(aio_path)
    aio_words = {r["query_id"]: content_words(clean_markdown(str(r["aio_text"])))
                 for _, r in aio_df.iterrows()}

    transcripts = {}
    for f in CACHE.glob("*.json"):
        try:
            transcripts[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

    real = pd.read_csv("data/raw/real.csv")
    real["domain"] = real["url"].str.extract(r"https?://([^/]+)/", expand=False)
    yt = real[real["domain"].str.contains("youtu", na=False) & (real["cited"] == 1)].copy()
    yt["video_id"] = yt["url"].map(video_id_from_url)
    yt["cited_at_s"] = yt["url"].map(timestamp_from_url)
    yt = yt[yt["video_id"].isin(transcripts) & yt["cited_at_s"].notna()]
    yt = yt.drop_duplicates(subset=["query_id", "video_id"])

    print("=" * 76)
    print("  DOES GOOGLE LINK TO THE PART OF THE VIDEO IT USED?")
    print("=" * 76)
    print(f"\n  Usable cases (transcript cached AND timestamp present): {len(yt)}")
    if len(yt) < 10:
        print("  Fewer than 10 cases - any result here is indicative at best.")

    rng = random.Random(SEED)
    all_qids = [q for q in aio_words if aio_words[q]]
    rows = []
    for _, r in yt.iterrows():
        tr = transcripts[r["video_id"]]
        wins = windows(tr.get("snippets", []), args.window, args.step)
        if len(wins) < 3:
            continue
        duration = max(w["start"] for w in wins) + args.window

        real_start, real_score = best_window(aio_words.get(r["query_id"], set()), wins)
        fake_qid = rng.choice([q for q in all_qids if q != r["query_id"]])
        fake_start, fake_score = best_window(aio_words[fake_qid], wins)

        rows.append({
            "query_id": r["query_id"], "query": r["query"], "video_id": r["video_id"],
            "cited_at_s": r["cited_at_s"], "duration_s": round(duration, 1),
            "best_match_s": real_start, "match_score": round(real_score, 3),
            "gap_s": abs(real_start - r["cited_at_s"]),
            "random_gap_s": abs(rng.uniform(0, duration) - r["cited_at_s"]),
            "unrelated_gap_s": abs(fake_start - r["cited_at_s"]),
            "unrelated_score": round(fake_score, 3),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No usable cases.")

    print(f"  Analysed: {len(df)} (search, video) pairs across "
          f"{df['video_id'].nunique()} videos")
    print(f"  Median video length: {df['duration_s'].median():.0f} s\n")

    print("-" * 76)
    print("  How far is the best-matching passage from the moment Google linked to?")
    print("-" * 76)
    print(f"\n  {'':34s} {'median gap':>12s} {'within 60s':>12s} {'within 90s':>12s}")
    for label, col in [("Actual AI Overview content", "gap_s"),
                       ("An unrelated AI Overview", "unrelated_gap_s"),
                       ("A random point in the video", "random_gap_s")]:
        print(f"  {label:34s} {df[col].median():10.0f} s "
              f"{(df[col] <= 60).mean():11.0%} {(df[col] <= 90).mean():12.0%}")

    real_w = (df["gap_s"] <= 60).mean()
    rand_w = (df["random_gap_s"] <= 60).mean()
    unrel_w = (df["unrelated_gap_s"] <= 60).mean()

    print("\n  Reading it:")
    if real_w > max(rand_w, unrel_w) + 0.15:
        print("    The passage matching the AI Overview sits closer to the linked moment")
        print("    than either baseline. Google located the relevant part of the video.")
    elif real_w > max(rand_w, unrel_w) + 0.05:
        print("    Slightly better than the baselines - suggestive, not conclusive at")
        print(f"    this sample size (n={len(df)}).")
    else:
        print("    No advantage over the baselines. On this data the timestamp cannot be")
        print("    shown to point at the content that was used. That is a real result:")
        print("    videos are cited, but reuse of their spoken content isn't demonstrated.")

    print(f"\n  Caveat: {sum(1 for t in transcripts.values() if t.get('is_generated'))} of "
          f"{len(transcripts)} transcripts are auto-generated, so their text is noisy.")
    print("  Word-overlap tolerates that better than exact matching, but not perfectly.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "video_alignment.csv", index=False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    top = max(df[["gap_s", "random_gap_s", "unrelated_gap_s"]].max())
    bins = np.linspace(0, top, 12)
    ax.hist(df["gap_s"], bins=bins, alpha=0.85, color="#4F46E5", label="Actual AI Overview")
    ax.hist(df["random_gap_s"], bins=bins, alpha=0.45, color="#94A3B8",
            label="Random point in the video")
    ax.set_xlabel("Distance from the moment Google linked to (seconds)")
    ax.set_ylabel("Number of cases")
    ax.set_title("Does Google point at the part of the video it used?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "case_6_video_alignment.png", dpi=130)
    plt.close(fig)

    print(f"\n  Table -> {RESULTS}/video_alignment.csv")
    print(f"  Chart -> {FIGURES}/case_6_video_alignment.png")

    print("\n  Closest cases:")
    for _, r in df.nsmallest(4, "gap_s").iterrows():
        print(f"    \"{r['query'][:46]}\"  linked {r['cited_at_s']:.0f}s, "
              f"best match {r['best_match_s']:.0f}s  (gap {r['gap_s']:.0f}s)")


if __name__ == "__main__":
    main()
