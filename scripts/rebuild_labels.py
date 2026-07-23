#!/usr/bin/env python3
"""Rebuild labels and ranks from the cache, fixing three defects at once.

What this repairs
-----------------
**1. The URL matching bug.** `collect/serp.py::normalize_url()` builds its key as
`scheme://host/path` — dropping the query string and keeping the scheme. Two
consequences:

  * every `youtube.com/watch?v=…` collapses to the same key, so in a dict of
    cited URLs only the last reference per search survived;
  * an `http://` organic result never matched an `https://` reference, producing
    false negatives (the same page appearing twice in one search: once ranked and
    `cited=0`, once as a rank-101 row and `cited=1`).

  Fixed here: scheme dropped, leading `www.` dropped, query string kept.

**2. `organic_rank` was the wrong field.** It holds DataForSEO's `rank_absolute`,
which counts *every* SERP block — including the AI Overview itself. So the
"rank" partly encodes which blocks are present, and block composition almost
determines whether a citation is possible at all. `rank_group` is the true
organic position and is in the cache. Both are written out.

**3. Labels that were wrong because of #1.** Recomputed from the raw references.

The content features are not touched — they come from the cached HTML, which
hasn't changed. Only ranks and labels are rebuilt, then joined back on
(query_id, url).

Output: `data/raw/real_v2.csv`, plus a report of exactly what changed. The
original file is left alone so results can be compared before and after.

    python scripts/rebuild_labels.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

SERP_DIR = Path("data/raw/_cache/serp_json")
RESULTS = Path("reports/results")


def normalize_url(url: str) -> str:
    """Canonical key for matching: no scheme, no leading www., query preserved.

    Keeping the query string is what stops different YouTube videos collapsing
    onto one key. Dropping the scheme is what makes http/https variants match.
    """
    if not isinstance(url, str) or not url.strip():
        return ""
    try:
        p = urlparse(url.strip())
    except ValueError:
        return url.strip().lower()
    host = (p.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (p.path or "").rstrip("/") or "/"
    query = f"?{p.query}" if p.query else ""
    return f"{host}{path}{query}"


def collect_references(items: list[dict]) -> list[str]:
    urls = []
    for it in items:
        if it.get("type") != "ai_overview":
            continue
        for ref in (it.get("references") or []):
            if ref.get("url"):
                urls.append(ref["url"])
        for el in (it.get("items") or []):
            for ref in (el.get("references") or []):
                if ref.get("url"):
                    urls.append(ref["url"])
    return urls


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/raw/real.csv")
    p.add_argument("--out", default="data/raw/real_v2.csv")
    args = p.parse_args()

    if not SERP_DIR.exists():
        raise SystemExit(f"Cache not found at {SERP_DIR}")

    old = pd.read_csv(args.data)
    print("=" * 76)
    print("  REBUILDING LABELS AND RANKS FROM THE CACHE")
    print(f"  Existing file: {len(old):,} rows, {old['query_id'].nunique()} searches")
    print("=" * 76)

    rebuilt = []
    n_files = 0
    for f in sorted(SERP_DIR.glob("*.json")):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            result = raw["tasks"][0]["result"][0]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue
        n_files += 1
        qid = f.stem
        items = result.get("items", [])
        cited_keys = {normalize_url(u) for u in collect_references(items)}

        for it in items:
            if it.get("type") != "organic":
                continue
            url = it.get("url")
            if not url:
                continue
            rebuilt.append({
                "query_id": qid,
                "url": url,
                "url_key": normalize_url(url),
                "rank_group_new": it.get("rank_group"),
                "rank_absolute_new": it.get("rank_absolute"),
                "cited_new": int(normalize_url(url) in cited_keys),
            })

    new = pd.DataFrame(rebuilt)
    print(f"\n  Parsed {n_files} cached SERP files -> {len(new):,} organic rows")

    # Join on the normalised key so http/https and www variants line up.
    old = old.copy()
    old["url_key"] = old["url"].map(normalize_url)
    merged = old.merge(
        new[["query_id", "url_key", "rank_group_new", "rank_absolute_new", "cited_new"]],
        on=["query_id", "url_key"], how="left",
    )
    merged = merged.drop_duplicates(subset=["query_id", "url_key"], keep="first")

    ranked_mask = merged["organic_rank"] != 101
    matched = merged["cited_new"].notna()
    print(f"  Matched to cached organic rows: {int((matched & ranked_mask).sum()):,} "
          f"of {int(ranked_mask.sum()):,} ranked rows "
          f"({(matched & ranked_mask).mean():.1%})")

    # ------------------------------------------------------------- what changed
    print("\n--- What changed --------------------------------------------------")
    comp = merged[matched & ranked_mask]
    label_changed = comp[comp["cited"] != comp["cited_new"]]
    print(f"\n  Labels corrected: {len(label_changed)}")
    if len(label_changed):
        to_1 = int((label_changed["cited_new"] == 1).sum())
        print(f"    0 -> 1 (was a missed citation): {to_1}")
        print(f"    1 -> 0:                         {len(label_changed) - to_1}")
        print("\n    Examples:")
        for _, r in label_changed.head(5).iterrows():
            print(f"      {str(r['query'])[:34]:36s} {str(r['url'])[:44]:46s} "
                  f"{int(r['cited'])} -> {int(r['cited_new'])}")

    rank_diff = comp[comp["organic_rank"] != comp["rank_group_new"]]
    print(f"\n  Rows where rank_group differs from the stored rank: {len(rank_diff)} "
          f"({len(rank_diff) / max(len(comp), 1):.0%})")
    if len(comp):
        print(f"    Median stored rank (rank_absolute): {comp['organic_rank'].median():.0f}")
        print(f"    Median true organic rank:           {comp['rank_group_new'].median():.0f}")
        shift = (comp["organic_rank"] - comp["rank_group_new"]).median()
        print(f"    Median shift:                       {shift:.0f} positions")
        print("\n    That shift is the SERP blocks above the organic results - which")
        print("    is exactly why the stored rank partly encoded whether an AI")
        print("    Overview was present.")

    # ------------------------------------------------------------ write the file
    out = merged.copy()
    out["organic_rank_absolute"] = out["rank_absolute_new"].fillna(out["organic_rank"])
    # Keep 101 as the sentinel for cited-but-not-ranked rows.
    out["organic_rank"] = out["rank_group_new"].fillna(101).astype(int)
    out["cited"] = out["cited_new"].fillna(out["cited"]).astype(int)
    out = out.drop(columns=["url_key", "rank_group_new", "rank_absolute_new", "cited_new"])

    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outpath, index=False)

    ranked_new = out[out["organic_rank"] != 101]
    print("\n--- The rebuilt file ----------------------------------------------")
    print(f"\n  Rows: {len(out):,} (ranked: {len(ranked_new):,})")
    print(f"  Citation rate, all rows:    {out['cited'].mean():.3f} "
          f"(was {old['cited'].mean():.3f})")
    print(f"  Citation rate, ranked rows: {ranked_new['cited'].mean():.3f} "
          f"(was {old[old['organic_rank'] != 101]['cited'].mean():.3f})")
    print(f"\n  Written to {outpath}")
    print("\n  The original file is untouched, so every result can be recomputed")
    print("  both ways. To use the rebuilt data:")
    print(f"    python scripts/run_definitive_analysis.py --data {outpath}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "rows": len(out),
        "ranked_rows": len(ranked_new),
        "labels_corrected": len(label_changed),
        "rank_rows_changed": len(rank_diff),
        "citation_rate_all_old": round(float(old["cited"].mean()), 4),
        "citation_rate_all_new": round(float(out["cited"].mean()), 4),
        "citation_rate_ranked_old": round(
            float(old[old["organic_rank"] != 101]["cited"].mean()), 4),
        "citation_rate_ranked_new": round(float(ranked_new["cited"].mean()), 4),
    }]).to_csv(RESULTS / "rebuild_report.csv", index=False)
    print(f"  Report -> {RESULTS}/rebuild_report.csv")


if __name__ == "__main__":
    main()
