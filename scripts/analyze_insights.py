#!/usr/bin/env python3
"""Turn the extracted cache data + real.csv into the 5 insight charts.

Run AFTER extract_from_cache.py:
    python scripts/analyze_insights.py --data data/raw/real.csv

Reads reports/results/*.csv (from extract_from_cache.py) and data/raw/real.csv,
writes charts to reports/figures/insight_*.png and prints the numbers behind each.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("reports/figures")
RESULTS = Path("reports/results")


def insight_1_local_vs_aio(meta: pd.DataFrame) -> None:
    """Local Pack vs AI Overview — the near-perfect mutual exclusion."""
    lp = meta[meta["has_local_pack"] == 1]["ai_overview_present"].mean()
    nolp = meta[meta["has_local_pack"] == 0]["ai_overview_present"].mean()
    print("\n[1] Local Pack vs AI Overview")
    print(f"    With local pack:    {lp:.1%} have an AI Overview (n={(meta['has_local_pack'] == 1).sum()})")
    print(f"    Without local pack: {nolp:.1%} have an AI Overview (n={(meta['has_local_pack'] == 0).sum()})")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Has Local Pack\n(local intent)", "No Local Pack"], [lp * 100, nolp * 100],
           color=["#d62728", "#2ca02c"])
    ax.set_ylabel("% of queries with an AI Overview")
    ax.set_title("Local intent and AI Overviews almost never co-occur")
    for i, v in enumerate([lp * 100, nolp * 100]):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(OUT / "insight_1_local_vs_aio.png", dpi=130)
    plt.close(fig)


def insight_2_serp_predictors(meta: pd.DataFrame) -> None:
    """Which SERP features predict AIO presence."""
    print("\n[2] SERP features as AIO predictors")
    feats = ["has_people_also_ask", "has_related_searches", "has_featured_snippet",
             "has_knowledge_graph", "has_local_pack"]
    data = []
    for f in feats:
        if f in meta.columns and (meta[f] == 1).sum() > 0:
            rate = meta[meta[f] == 1]["ai_overview_present"].mean()
            data.append((f.replace("has_", ""), rate * 100, (meta[f] == 1).sum()))
            print(f"    {f:24s}: {rate:.0%} AIO-rate (n={(meta[f] == 1).sum()})")
    data.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    ax.barh(labels, vals, color="#1f77b4")
    ax.set_xlabel("% of queries with an AI Overview when this feature is present")
    ax.set_title("Some SERP features strongly signal an AI Overview")
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.0f}%", va="center")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(OUT / "insight_2_serp_predictors.png", dpi=130)
    plt.close(fig)


def insight_3_top_domains(df: pd.DataFrame) -> None:
    """Most-cited domains + the YouTube anomaly."""
    print("\n[3] Most-cited domains")
    df = df.copy()
    df["domain"] = df["url"].str.extract(r"https?://([^/]+)/", expand=False).str.replace("www.", "", regex=False)
    cited = df[df["cited"] == 1]
    top = cited["domain"].value_counts().head(12)
    for dom, n in top.items():
        total = (df["domain"] == dom).sum()
        print(f"    {dom:35s} {n:4d} cites / {total:4d} seen ({n / total:.0%})")

    fig, ax = plt.subplots(figsize=(8, 5))
    top_sorted = top.sort_values()
    ax.barh(top_sorted.index, top_sorted.values, color="#ff7f0e")
    ax.set_xlabel("Number of AI Overview citations")
    ax.set_title("Which domains AI Overviews cite most (real-estate queries)")
    fig.tight_layout()
    fig.savefig(OUT / "insight_3_top_domains.png", dpi=130)
    plt.close(fig)


def insight_4_source_count(meta: pd.DataFrame) -> None:
    """How many sources an AIO cites."""
    aio = meta[meta["ai_overview_present"] == 1]
    col = None
    for candidate in ("aio_num_references", "ai_overview_num_references"):
        if candidate in aio.columns:
            col = candidate
            break
    if col is None:
        print("\n[4] (aio_num_references not in meta — skipping chart)")
        return
    print("\n[4] Sources cited per AI Overview")
    print(f"    Median: {aio[col].median():.0f}, mean: {aio[col].mean():.1f}, "
          f"range: {aio[col].min():.0f}-{aio[col].max():.0f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(aio[col], bins=20, color="#9467bd", edgecolor="white")
    ax.axvline(aio[col].median(), color="black", linestyle="--", label=f"median = {aio[col].median():.0f}")
    ax.set_xlabel("Number of sources cited in one AI Overview")
    ax.set_ylabel("Number of queries")
    ax.set_title("AI Overviews cite many sources, not 'a handful'")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "insight_4_source_count.png", dpi=130)
    plt.close(fig)


def insight_5_wordcount_curve(df: pd.DataFrame) -> None:
    """The U-shaped word-count relationship."""
    print("\n[5] Word count vs citation rate (U-shape)")
    df = df.copy()
    df["wc_bucket"] = pd.cut(df["word_count"], bins=[0, 500, 1000, 2000, 4000, 100000],
                             labels=["<500", "500-1k", "1k-2k", "2k-4k", ">4k"])
    grp = df.groupby("wc_bucket", observed=True)["cited"].agg(["mean", "size"])
    for bucket, row in grp.iterrows():
        print(f"    {bucket:8s}: {row['mean']:.1%} cited (n={int(row['size'])})")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(grp)), grp["mean"] * 100, marker="o", linewidth=2, markersize=9, color="#d62728")
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels(grp.index)
    ax.set_xlabel("Word count of the page")
    ax.set_ylabel("% of pages cited")
    ax.set_title("Citation rate is U-shaped, not linear\n(short precise answers AND long deep pages win)")
    for i, v in enumerate(grp["mean"] * 100):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "insight_5_wordcount_curve.png", dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=str, default="data/raw/real.csv")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)

    # Prefer the enriched meta from the cache; fall back to real_meta.csv.
    enriched = RESULTS / "serp_features_enriched.csv"
    meta_path = enriched if enriched.exists() else Path(args.data).with_name("real_meta.csv")
    meta = pd.read_csv(meta_path)
    print(f"Using SERP meta: {meta_path}")

    insight_1_local_vs_aio(meta)
    insight_2_serp_predictors(meta)
    insight_3_top_domains(df)
    insight_4_source_count(meta)
    insight_5_wordcount_curve(df)

    print(f"\nAll 5 insight charts saved to {OUT}/insight_*.png")


if __name__ == "__main__":
    main()
