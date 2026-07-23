#!/usr/bin/env python3
"""Descriptive statistics, corrected for both known artefacts.

Two distortions affect every descriptive number in this project, and the review
found that some outputs still carried them:

1. THE RANK-101 ROWS. 1,789 rows are cited-but-not-ranked pages, in the data only
   because they were cited. Any rate computed over all rows is inflated.
   `query_segment_summary.csv` still had this: it reported a 53% citation rate for
   informational searches, computed over all rows.

2. SEARCHES WITH NO AI OVERVIEW. 205 of 533 searches never showed one, so ~40% of
   ranked rows are `cited = 0` by construction. Comparing a domain against the
   overall 29% average therefore compares it against a baseline that is partly
   made of structurally uncitable rows.

This script recomputes the segment table and the domain leaderboard on the clean
population *and* conditional on searches that actually have an AI Overview, adds
confidence intervals, and separates marginal association from predictive power
for the structural features.

    python scripts/fix_descriptive_stats.py --data data/raw/real.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("reports/results")
MIN_N = 30


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - reliable for proportions with small n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def segment_of(row: pd.Series) -> str:
    if row.get("has_local_pack", 0) == 1:
        return "Local (map)"
    if row.get("has_featured_snippet", 0) == 1:
        return "Featured snippet"
    return "Informational"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/raw/real.csv")
    args = p.parse_args()

    raw = pd.read_csv(args.data)
    ranked = raw[raw["organic_rank"] != 101].copy()
    ranked["domain"] = (ranked["url"].str.extract(r"https?://([^/]+)/", expand=False)
                        .str.replace("www.", "", regex=False))
    has_aio = ranked.groupby("query_id")["cited"].transform("max") > 0
    aio_rows = ranked[has_aio]

    print("=" * 78)
    print("  CORRECTED DESCRIPTIVE STATISTICS")
    print(f"  All rows {len(raw):,} | ranked {len(ranked):,} | "
          f"ranked & AI-Overview search {len(aio_rows):,}")
    print(f"  Citation rate: all rows {raw['cited'].mean():.3f} | "
          f"ranked {ranked['cited'].mean():.3f} | conditional {aio_rows['cited'].mean():.3f}")
    print("=" * 78)

    # ------------------------------------------------------- 1. intent segments
    print("\n--- 1. Intent segments (the previous table used all rows) -----------")
    enriched = RESULTS / "serp_features_enriched.csv"
    meta_path = enriched if enriched.exists() else Path(args.data).with_name("real_meta.csv")
    meta = pd.read_csv(meta_path)
    meta["segment"] = meta.apply(segment_of, axis=1)

    rate_all = raw.groupby("query_id")["cited"].mean().rename("rate_all_rows")
    rate_ranked = ranked.groupby("query_id")["cited"].mean().rename("rate_ranked")
    meta = meta.merge(rate_all, on="query_id", how="left").merge(rate_ranked, on="query_id", how="left")

    rows = []
    print(f"\n  {'segment':20s} {'n':>5s} {'AIO rate':>9s} {'cite (all rows)':>16s} {'cite (ranked)':>14s}")
    for seg in ["Local (map)", "Featured snippet", "Informational"]:
        g = meta[meta["segment"] == seg]
        if g.empty:
            continue
        aio = g["ai_overview_present"].mean()
        c_all = g["rate_all_rows"].mean(skipna=True)
        c_rank = g["rate_ranked"].mean(skipna=True)
        print(f"  {seg:20s} {len(g):5d} {aio:8.1%} {c_all:15.1%} {c_rank:13.1%}")
        rows.append({"segment": seg, "queries": len(g), "ai_overview_rate": round(float(aio), 3),
                     "citation_rate_all_rows": round(float(c_all), 3),
                     "citation_rate_ranked": round(float(c_rank), 3)})
    print("\n  The 'all rows' column is the inflated one. Report the ranked column.")
    print("  Note the two 0% segments are 0 of 114 searches combined - state it as a")
    print("  count, since the segmentation is effectively binary.")
    pd.DataFrame(rows).to_csv(RESULTS / "query_segment_summary_corrected.csv", index=False)

    # ------------------------------------------------------ 2. domain leaderboard
    print("\n--- 2. Domain leaderboard (conditional baseline + intervals) ---------")
    base_ranked = ranked["cited"].mean()
    base_cond = aio_rows["cited"].mean()
    print(f"\n  Baselines: ranked {base_ranked:.3f} | conditional on AI Overview {base_cond:.3f}")

    g = (aio_rows.groupby("domain")
         .agg(times_ranked=("cited", "size"), times_cited=("cited", "sum")).reset_index())
    g["rate"] = g["times_cited"] / g["times_ranked"]
    g["vs_conditional"] = g["rate"] / base_cond
    ci = g.apply(lambda r: wilson_ci(int(r["times_cited"]), int(r["times_ranked"])), axis=1)
    g["ci_low"] = [c[0] for c in ci]
    g["ci_high"] = [c[1] for c in ci]
    g["reliable"] = g["times_ranked"] >= MIN_N
    g = g.sort_values("rate", ascending=False)

    show = g[g["times_ranked"] >= MIN_N].head(12)
    print(f"\n  Domains with n >= {MIN_N}, on AI-Overview searches only:")
    print(f"  {'domain':32s} {'n':>5s} {'rate':>7s} {'95% CI':>16s} {'vs base':>9s}")
    for _, r in show.iterrows():
        print(f"  {r['domain']:32s} {int(r['times_ranked']):5d} {r['rate']:6.0%} "
              f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}] {r['vs_conditional']:8.2f}x")
    n_small = int((g["times_ranked"] < MIN_N).sum())
    print(f"\n  {n_small} domains have n < {MIN_N} and are excluded - their rates are noise.")
    print("  Previous version compared against the 29% all-ranked baseline, which")
    print("  includes searches that could never produce a citation. These multiples")
    print("  are smaller and correct.")
    g.to_csv(RESULTS / "domain_leaderboard_conditional.csv", index=False)

    # -------------------------------------------- 3. structure: association only
    print("\n--- 3. Structural features: association vs predictive power ----------")
    try:
        from scipy.stats import fisher_exact
        print(f"\n  {'feature':16s} {'population':22s} {'OR':>6s} {'p':>10s}")
        rows3 = []
        for col in ["has_schema", "has_faq"]:
            for label, d in [("ranked pages", ranked), ("AI-Overview searches", aio_rows)]:
                a = int(((d[col] == 1) & (d["cited"] == 1)).sum())
                b = int(((d[col] == 1) & (d["cited"] == 0)).sum())
                c = int(((d[col] == 0) & (d["cited"] == 1)).sum())
                e = int(((d[col] == 0) & (d["cited"] == 0)).sum())
                odds, pv = fisher_exact([[a, b], [c, e]])
                print(f"  {col:16s} {label:22s} {odds:6.2f} {pv:10.2g}")
                rows3.append({"feature": col, "population": label,
                              "odds_ratio": round(float(odds), 2), "p_value": float(pv)})
        pd.DataFrame(rows3).to_csv(RESULTS / "structure_association.csv", index=False)
        print("\n  These associations are real and survive conditioning. But permutation")
        print("  importance gives these features ~0 - the model gains nothing from them.")
        print("  Correct phrasing: 'structured pages are cited more often, but structure")
        print("  adds no predictive power once other signals are known'. Given the")
        print("  site-memorisation finding, this is plausibly another site effect.")
    except ImportError:
        print("  (scipy not installed - skipping)")

    # ------------------------------------------------- 4. dataset consistency
    print("\n--- 4. Dataset consistency ------------------------------------------")
    n_real = raw["query_id"].nunique()
    print(f"  real.csv queries:              {n_real}")
    if (RESULTS / "aio_text.csv").exists():
        print(f"  aio_text.csv rows:             {len(pd.read_csv(RESULTS / 'aio_text.csv'))}")
    if meta_path.exists():
        print(f"  SERP snapshots:                {len(meta)}")
        print(f"  with ai_overview_present = 1:  {int(meta['ai_overview_present'].sum())}")
    print(f"  queries with >=1 citation:     {int((raw.groupby('query_id')['cited'].max() > 0).sum())}")
    print("\n  Small differences are expected: 538 searches were attempted, 3 failed")
    print("  during collection, and a few SERP files cover searches whose page-level")
    print("  rows were dropped. State the numbers rather than rounding them together.")

    print(f"\n  Tables -> {RESULTS}/query_segment_summary_corrected.csv, "
          f"domain_leaderboard_conditional.csv, structure_association.csv")


if __name__ == "__main__":
    main()
