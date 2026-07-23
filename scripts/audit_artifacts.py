#!/usr/bin/env python3
"""Artifact audit - re-check every descriptive finding on the clean population.

Why this exists
---------------
The dataset contains 1,789 "rank-101" rows: pages the AI Overview cited that
never appeared in Google's visible results. They are in the data *only because*
they were cited, so every one of them has ``cited = 1`` by construction.

Any descriptive statistic computed over all rows is therefore distorted wherever
those rows cluster. Two published findings turned out to be exactly that:

  * "YouTube is cited 95% of the time" - 92% of its rows were rank-101. Among
    YouTube pages that genuinely rank, the rate is 36%.
  * "Citation rate is U-shaped in word count" - the short-page peak was made
    entirely of rank-101 rows. On ranked pages the relationship rises with
    length instead.

This script recomputes each descriptive finding both ways so the distortion is
visible rather than assumed. Anything where the two columns diverge should be
reported on the ranked-page population only.

Query-level findings (intent segments, SERP features, sources per answer) are
not affected: rank-101 is a page-level artefact, and those statistics count
queries.

    python scripts/audit_artifacts.py --data data/raw/real.csv
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


def compare(all_rows: pd.DataFrame, ranked: pd.DataFrame, mask_col: str, label: str) -> dict:
    """Citation rate for a binary flag, computed both ways.

    The comparison is *lift over the population baseline*, not the raw rate: the
    baseline itself shifts (48% -> 29%) when the always-cited rows are removed,
    so raw rates always look different. What matters is whether the group still
    stands out from its own population by the same amount.
    """
    a = all_rows[all_rows[mask_col] == 1]["cited"].mean() if (all_rows[mask_col] == 1).any() else float("nan")
    r = ranked[ranked[mask_col] == 1]["cited"].mean() if (ranked[mask_col] == 1).any() else float("nan")
    lift_a = a / all_rows["cited"].mean()
    lift_r = r / ranked["cited"].mean()
    n_r = int((ranked[mask_col] == 1).sum())
    # Flag when the group's standing relative to its population materially changes.
    distorted = bool(abs(lift_a - lift_r) > 0.30 or (lift_a - 1) * (lift_r - 1) < 0)
    return {"finding": label, "all_rows": round(a, 3), "ranked_only": round(r, 3),
            "lift_all": round(lift_a, 2), "lift_ranked": round(lift_r, 2),
            "n_ranked": n_r, "distorted": distorted}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/raw/real.csv")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    df["is_101"] = (df["organic_rank"] == 101).astype(int)
    ranked = df[df["is_101"] == 0].copy()

    print("=" * 74)
    print("  ARTIFACT AUDIT")
    print(f"  All rows: {len(df):,} ({df['cited'].mean():.1%} cited)   "
          f"Ranked only: {len(ranked):,} ({ranked['cited'].mean():.1%} cited)")
    print(f"  rank-101 rows removed: {int(df['is_101'].sum()):,} - all cited by construction")
    print("=" * 74)

    # ---- Binary page flags
    rows = []
    for col, label in [("is_video", "Video pages"), ("is_forum", "Forum pages"),
                       ("has_schema", "Has schema markup"), ("has_faq", "Has FAQ section"),
                       ("is_https", "Uses HTTPS")]:
        if col in df.columns:
            rows.append(compare(df, ranked, col, label))

    print(f"\n  Lift = how far the group stands out from its own population's baseline.")
    print(f"  {'finding':26s} {'lift all':>9s} {'lift ranked':>12s} {'n':>7s}   verdict")
    for r in rows:
        verdict = "DISTORTED" if r["distorted"] else "ok"
        print(f"  {r['finding']:26s} {r['lift_all']:8.2f}x {r['lift_ranked']:11.2f}x "
              f"{r['n_ranked']:7d}   {verdict}")

    # ---- Word-count curve, both ways
    def curve(d):
        d = d.copy()
        d["b"] = pd.cut(d["word_count"], bins=[0, 500, 1000, 2000, 4000, 100000],
                        labels=["<500", "500-1k", "1k-2k", "2k-4k", ">4k"])
        return d.groupby("b", observed=True)["cited"].agg(["mean", "size"])

    c_all, c_ranked = curve(df), curve(ranked)
    print(f"\n  Word count vs citation rate")
    print(f"  {'bucket':10s} {'all rows':>10s} {'ranked only':>13s} {'n ranked':>10s}")
    for b in c_ranked.index:
        a = c_all.loc[b, "mean"] if b in c_all.index else float("nan")
        r = c_ranked.loc[b, "mean"]
        print(f"  {b:10s} {a:9.1%} {r:12.1%} {int(c_ranked.loc[b, 'size']):10d}")
    print("  -> the short-page peak in the all-rows column is the artefact;")
    print("     on ranked pages citation rate rises with length and plateaus.")

    # ---- Where do the rank-101 rows concentrate?
    print(f"\n  Where the rank-101 rows cluster (share of each group that is rank-101):")
    for col, label in [("is_video", "Video pages"), ("is_forum", "Forum pages")]:
        if col in df.columns and (df[col] == 1).any():
            share = df[df[col] == 1]["is_101"].mean()
            print(f"    {label:22s} {share:.0%}")
    short = df[df["word_count"] < 500]
    print(f"    {'Pages under 500 words':22s} {short['is_101'].mean():.0%}")

    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "artifact_audit.csv", index=False)

    # ---- Corrected word-count chart (replaces the U-curve figure)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(c_ranked))
    ax.plot(x, c_ranked["mean"] * 100, marker="o", linewidth=2.5, markersize=9,
            color="#4F46E5", label="Ranked pages (correct)")
    ax.plot(x, [c_all.loc[b, "mean"] * 100 for b in c_ranked.index], marker="o",
            linewidth=1.5, markersize=6, linestyle="--", color="#94A3B8",
            label="All rows (distorted by selection)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(c_ranked.index)
    ax.set_xlabel("Word count of the page")
    ax.set_ylabel("% of pages cited")
    ax.set_title("More content is cited more often\n(the earlier U-shape was a selection artefact)")
    ax.legend()
    for i, v in enumerate(c_ranked["mean"] * 100):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "insight_5_wordcount_curve.png", dpi=130)
    plt.close(fig)

    print(f"\n  Table -> {RESULTS}/artifact_audit.csv")
    print(f"  Chart -> {FIGURES}/insight_5_wordcount_curve.png  (corrected, shows both lines)")
    print("\n  Not affected (query-level statistics, not page-level):")
    print("    intent segments, SERP-feature predictors, sources per AI Overview,")
    print("    and the passage analysis (its control group is sentences from the")
    print("    same pages, so selection into the dataset cancels out).")


if __name__ == "__main__":
    main()
