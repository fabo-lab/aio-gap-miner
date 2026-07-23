#!/usr/bin/env python3
"""Analysis 3 - segment queries by search intent, reveal citation behaviour.

Not every search even gets an AI Overview, so not every search is worth
optimising for one. This segments queries by the signal Google shows for them
and reveals, per segment, how often an AI Overview appears and how often pages
get cited. It's the narrative backbone: it tells you *which searches* to compete
for in the first place.

Method note: an exploratory KMeans clustering was tried first, but on these
sparse SERP-feature flags it produced noisy, overlapping groups. A transparent
rule-based segmentation on Google's own dominant signals (local pack / featured
snippet / neither) is clearer, fully explainable, and - as the output shows -
separates the citation outcome almost perfectly. Choosing the interpretable tool
over the fancy one is the point.

Segments are defined by SERP signals only (never by the citation outcome), so
the "which segment gets cited" finding is not circular.

    python scripts/analyze_query_intent.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")
SEGMENT_ORDER = ["Local (map)", "Featured snippet", "Informational"]


def segment_of(row: pd.Series) -> str:
    """Assign a query to an intent segment by Google's dominant SERP signal."""
    if row.get("has_local_pack", 0) == 1:
        return "Local (map)"
    if row.get("has_featured_snippet", 0) == 1:
        return "Featured snippet"
    return "Informational"


def main() -> None:
    enriched = RESULTS / "serp_features_enriched.csv"
    meta_path = enriched if enriched.exists() else Path("data/raw/real_meta.csv")
    meta = pd.read_csv(meta_path)
    print(f"Using SERP meta: {meta_path}  ({len(meta)} queries)")

    # Citation rate per query, computed from the page-level dataset (robust: the
    # enriched SERP file doesn't carry candidate counts, real.csv always does).
    real_path = Path("data/raw/real.csv")
    if real_path.exists():
        real = pd.read_csv(real_path)
        rate = real.groupby("query_id")["cited"].mean().rename("citation_rate")
        meta = meta.merge(rate, on="query_id", how="left")
    elif {"num_cited", "num_candidates"}.issubset(meta.columns):
        meta["citation_rate"] = meta["num_cited"] / meta["num_candidates"].clip(lower=1)
    else:
        meta["citation_rate"] = float("nan")
    meta["citation_rate"] = meta["citation_rate"].fillna(0.0)

    meta["segment"] = meta.apply(segment_of, axis=1)

    rows = []
    for seg in SEGMENT_ORDER:
        grp = meta[meta["segment"] == seg]
        if grp.empty:
            continue
        rows.append({
            "segment": seg,
            "queries": len(grp),
            "pct_of_queries": round(len(grp) / len(meta), 3),
            "ai_overview_rate": round(grp["ai_overview_present"].mean(), 3),
            "citation_rate": round(grp["citation_rate"].mean(), 3),
            "example_queries": " | ".join(grp["query"].head(4).astype(str).tolist()),
        })
    summary = pd.DataFrame(rows)

    RESULTS.mkdir(parents=True, exist_ok=True)
    meta[["query_id", "query", "segment", "ai_overview_present", "citation_rate"]].to_csv(
        RESULTS / "query_segments.csv", index=False
    )
    summary.to_csv(RESULTS / "query_segment_summary.csv", index=False)

    print("\n  query_segments.csv          (per query)")
    print("  query_segment_summary.csv   (per segment)\n")
    print("Intent segments (Google's signal -> is AIO optimisation even worth it?):")
    for _, r in summary.iterrows():
        print(f"\n  [{r['segment']}]  - {r['queries']} queries ({r['pct_of_queries']:.0%})")
        print(f"     AI Overview appears: {r['ai_overview_rate']:.0%}   |   "
              f"citation rate: {r['citation_rate']:.0%}")
        print(f"     e.g. {r['example_queries'][:95]}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(summary))
    w = 0.38
    ax.bar([i - w / 2 for i in x], summary["ai_overview_rate"] * 100, width=w,
           label="AI Overview appears", color="#2ca02c")
    ax.bar([i + w / 2 for i in x], summary["citation_rate"] * 100, width=w,
           label="Citation rate", color="#1f77b4")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r['segment']}\n(n={r['queries']}, {r['pct_of_queries']:.0%})"
                        for _, r in summary.iterrows()])
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_title("Search intent decides whether AI Overview optimisation is even worth it")
    ax.legend()
    for i, r in enumerate(summary.itertuples()):
        ax.text(i - w / 2, r.ai_overview_rate * 100 + 2, f"{r.ai_overview_rate:.0%}", ha="center", fontsize=9)
        ax.text(i + w / 2, r.citation_rate * 100 + 2, f"{r.citation_rate:.0%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "insight_8_intent_segments.png", dpi=130)
    plt.close(fig)
    print(f"\nChart -> {FIGURES}/insight_8_intent_segments.png")


if __name__ == "__main__":
    main()
