#!/usr/bin/env python3
"""Collect real (query, URL) AI Overview citation data into the Gap-Miner schema.

Setup (once):
    export DATAFORSEO_LOGIN="your_login"
    export DATAFORSEO_PASSWORD="your_password"
    # optional, for real authority features:
    # export MOZ_TOKEN="your_moz_token"

Dry run (no credentials, no network -- proves the pipeline end to end):
    python scripts/collect_real_data.py --dry-run --out data/raw/dryrun.csv

Real run (Germany by default):
    python scripts/collect_real_data.py --queries queries.txt --out data/raw/real.csv

Then train on it with no code changes:
    python scripts/run_pipeline.py --data data/raw/real.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aio_gap_miner.collect import build_dataset

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "serp_sample.json"


def read_queries(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--queries", type=str, help="Text file, one query per line.")
    p.add_argument("--out", type=str, default="data/raw/real_citations.csv")
    p.add_argument("--location-code", type=int, default=2276, help="2276=DE, 2840=US.")
    p.add_argument("--language-code", type=str, default="de")
    p.add_argument(
        "--max-organic",
        type=int,
        default=15,
        help="Top-N organic candidates to keep per query (cited URLs always kept).",
    )
    p.add_argument("--limit", type=int, default=None, help="Only process the first N queries.")
    p.add_argument("--delay", type=float, default=1.0, help="Seconds between page crawls.")
    p.add_argument(
        "--no-crawl",
        action="store_true",
        help="Skip page crawl; use SERP snippets only (features imputed).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run offline against the bundled SERP fixture (no API, no network).",
    )
    args = p.parse_args()

    if args.dry_run:
        queries = ["how to descale an espresso machine"]
        fixture = str(FIXTURE)
        crawl = False
        print("DRY RUN: bundled fixture, no network. Proves schema + assembly.\n")
    else:
        if not args.queries:
            p.error("--queries is required unless --dry-run is set.")
        queries = read_queries(args.queries)
        fixture = None
        crawl = not args.no_crawl

    if args.limit:
        queries = queries[: args.limit]

    df = build_dataset(
        queries,
        location_code=args.location_code,
        language_code=args.language_code,
        max_organic=args.max_organic,
        crawl=crawl,
        fixture=fixture,
        polite_delay=args.delay,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    pos = int(df["cited"].sum())
    print(
        f"\nWrote {len(df):,} rows x {df.shape[1]} cols across "
        f"{df['query_id'].nunique()} queries -> {out}"
    )
    print(f"Cited (positives): {pos:,} ({pos / len(df):.1%})")
    aio_queries = df.groupby("query_id")["cited"].max().sum()
    print(f"Queries with at least one citation: {int(aio_queries)}/{df['query_id'].nunique()}")


if __name__ == "__main__":
    main()
