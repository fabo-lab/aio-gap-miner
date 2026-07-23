#!/usr/bin/env python3
"""Corrected domain analysis - which sites get cited when they actually rank?

Why this replaces the earlier version
-------------------------------------
The first domain leaderboard computed citation rate over ALL rows, including the
rank-101 sentinel rows (cited pages that never appeared in Google's visible
results). Those rows are in the dataset *only because* they were cited, so any
domain that mostly appears through them shows a near-100% "citation rate" that
measures nothing.

YouTube was the clearest case: 330 rows, 95% cited - but 305 of those (92%) were
rank-101. Among YouTube pages that genuinely rank, only 36% get cited. In fact,
across all video pages that rank, the citation rate is 15.8% versus 29.3% for
non-video pages: the opposite of the original conclusion.

This script restricts to genuinely ranked pages, where the question is well
posed: when this domain appears in Google's results, how often does the AI
Overview cite it?

    python scripts/analyze_domains.py --data data/raw/real.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")
MIN_APPEARANCES = 20


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/raw/real.csv")
    p.add_argument("--min-appearances", type=int, default=MIN_APPEARANCES)
    args = p.parse_args()

    raw = pd.read_csv(args.data)
    raw["domain"] = (
        raw["url"].str.extract(r"https?://([^/]+)/", expand=False)
        .str.replace("www.", "", regex=False)
    )
    ranked = raw[raw["organic_rank"] != 101].copy()
    overall = ranked["cited"].mean()

    print(f"Ranked pages only: {len(ranked):,} rows (of {len(raw):,})")
    print(f"Overall citation rate among ranked pages: {overall:.1%}\n")

    g = (
        ranked.groupby("domain")
        .agg(times_ranked=("cited", "size"), times_cited=("cited", "sum"))
        .reset_index()
    )
    g["citation_rate"] = (g["times_cited"] / g["times_ranked"]).round(3)
    g["vs_average"] = (g["citation_rate"] / overall).round(2)
    g = g[g["times_ranked"] >= args.min_appearances].sort_values("citation_rate", ascending=False)

    print(f"Domains appearing at least {args.min_appearances} times, by citation rate:\n")
    print(f"  {'domain':34s} {'ranked':>7s} {'cited':>7s} {'rate':>7s} {'vs avg':>8s}")
    for _, r in g.head(15).iterrows():
        print(f"  {r['domain']:34s} {int(r['times_ranked']):7d} {int(r['times_cited']):7d} "
              f"{r['citation_rate']:6.0%} {r['vs_average']:7.2f}x")

    # The video correction, stated explicitly - it's a finding in itself.
    print("\nVideo pages (the corrected picture):")
    for label, sub in [("video, ranked", ranked[ranked["is_video"] == 1]),
                       ("non-video, ranked", ranked[ranked["is_video"] == 0])]:
        if len(sub):
            print(f"  {label:20s} n={len(sub):5d}  cited {sub['cited'].mean():.1%}")
    all_video = raw[raw["is_video"] == 1]
    if len(all_video):
        share_101 = (all_video["organic_rank"] == 101).mean()
        print(f"  -> {share_101:.0%} of all video rows are cited-but-not-ranked, which is why the")
        print("     uncorrected number looked so high. Video content enters AI Overviews")
        print("     mostly from outside the visible results, not by ranking well.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    g.to_csv(RESULTS / "domain_citation_ranked_only.csv", index=False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    top = g.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top["domain"], top["citation_rate"] * 100, color="#4F46E5")
    ax.axvline(overall * 100, color="#94A3B8", linestyle="--", linewidth=1.5,
               label=f"average ({overall:.0%})")
    ax.set_xlabel("% of the time this domain is cited when it ranks")
    ax.set_title("Trusted brands get cited about twice as often as average")
    ax.legend()
    for i, v in enumerate(top["citation_rate"] * 100):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "insight_3_top_domains.png", dpi=130)
    plt.close(fig)

    print(f"\nTable -> {RESULTS}/domain_citation_ranked_only.csv")
    print(f"Chart -> {FIGURES}/insight_3_top_domains.png  (replaces the earlier version)")


if __name__ == "__main__":
    main()
