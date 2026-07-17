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

Two outputs are written for a real run:
  <out>                          -- one row per (query, URL) candidate (the ML training set)
  <out with _meta suffix>        -- one row per query: SERP feature flags (local_pack,
                                     people_also_ask, ...) and AI Overview stats, for
                                     clustering queries by intent (local/informational/
                                     transactional).

For long batches, rows are written incrementally after every query (a raw,
pre-finalised file next to <out>), so a crash or interruption partway through
never loses already-collected progress. A query whose SERP fetch fails (network
blip, transient error) is logged and skipped -- it does not abort the whole run.

By default, the raw DataForSEO JSON response per query and the raw HTML of
every crawled page are also cached to disk under <out's folder>/_cache/ (git-
ignored, local only). This is a permanent snapshot: SERPs change over time, so
without this, a future re-analysis with different features or questions would
need a fresh (and different) query rather than reproducing tonight's exact
data. Disable with --no-cache if disk space is a concern.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Load credentials from a local, git-ignored .env if python-dotenv is installed.
# Falls back silently to real environment variables (e.g. set in your shell).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from aio_gap_miner.collect import DataForSEOClient, build_dataset, collect_query, finalise_dataset

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "serp_sample.json"


def read_queries(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def meta_path_for(out: Path) -> Path:
    return out.with_name(out.stem + "_meta" + out.suffix)


def raw_path_for(out: Path) -> Path:
    return out.with_name(out.stem + "_raw" + out.suffix)


def cache_dir_for(out: Path) -> Path:
    return out.parent / "_cache"


def run_incremental(
    queries: list[str],
    *,
    out: Path,
    location_code: int,
    language_code: str,
    max_organic: int,
    crawl: bool,
    delay: float,
    cache_dir: Path | None,
) -> None:
    """Crash-tolerant collection: write every query's rows to disk immediately,
    skip (and report) any query whose SERP fetch fails, and only run the global
    finalise step (domain_citation_rate, imputation) once at the very end.
    """
    client = DataForSEOClient()
    raw_out = raw_path_for(out)
    meta_out = meta_path_for(out)
    raw_out.parent.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    n_rows = 0
    header_written = False
    meta_header_written = False

    for i, q in enumerate(queries):
        print(f"[{i + 1}/{len(queries)}] {q!r}", flush=True)
        try:
            rows, meta = collect_query(
                q,
                f"q{i:04d}",
                client,
                location_code=location_code,
                language_code=language_code,
                max_organic=max_organic,
                crawl=crawl,
                fixture=None,
                polite_delay=delay,
                cache_dir=cache_dir,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad query must not kill an 8h run
            print(f"    SKIPPED (error: {exc})", flush=True)
            failed.append(q)
            continue

        pd.DataFrame(rows).to_csv(raw_out, mode="a", index=False, header=not header_written)
        header_written = True
        n_rows += len(rows)

        pd.DataFrame([meta]).to_csv(meta_out, mode="a", index=False, header=not meta_header_written)
        meta_header_written = True

    if n_rows == 0:
        raise RuntimeError("No rows collected at all -- check credentials/queries.")

    print(f"\nRaw collection done: {n_rows:,} rows -> {raw_out}")
    if failed:
        print(f"{len(failed)} quer{'y' if len(failed) == 1 else 'ies'} skipped after errors:")
        for q in failed:
            print(f"    - {q}")
        print("Re-run just these later with a small --queries file if you want full coverage.")

    print("Applying global finalise step (domain_citation_rate, imputation) ...")
    raw_df = pd.read_csv(raw_out)
    final_df = finalise_dataset(raw_df)
    final_df.to_csv(out, index=False)

    pos = int(final_df["cited"].sum())
    print(
        f"\nWrote {len(final_df):,} rows x {final_df.shape[1]} cols across "
        f"{final_df['query_id'].nunique()} queries -> {out}"
    )
    print(f"Cited (positives): {pos:,} ({pos / len(final_df):.1%})")
    aio_q = final_df.groupby("query_id")["cited"].max().sum()
    print(f"Queries with >=1 citation: {int(aio_q)}/{final_df['query_id'].nunique()}")
    print(f"Query-level SERP-feature metadata -> {meta_out}")
    if cache_dir is not None:
        print(f"Raw SERP JSON + HTML cached under -> {cache_dir}  (git-ignored, local only)")


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
        "--no-cache",
        action="store_true",
        help="Skip caching raw SERP JSON + HTML to disk (on by default; saves a "
        "permanent local snapshot for future re-analysis without re-querying/re-crawling).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run offline against the bundled SERP fixture (no API, no network).",
    )
    args = p.parse_args()

    if args.dry_run:
        queries = ["how to descale an espresso machine"]
        print("DRY RUN: bundled fixture, no network. Proves schema + assembly.\n")
        df, meta_df = build_dataset(queries, crawl=False, fixture=str(FIXTURE), verbose=True)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        meta_df.to_csv(meta_path_for(out), index=False)
        pos = int(df["cited"].sum())
        print(
            f"\nWrote {len(df):,} rows x {df.shape[1]} cols across "
            f"{df['query_id'].nunique()} queries -> {out}"
        )
        print(f"Cited (positives): {pos:,} ({pos / len(df):.1%})")
        print(f"Query metadata -> {meta_path_for(out)}")
        return

    if not args.queries:
        p.error("--queries is required unless --dry-run is set.")
    queries = read_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]

    out_path = Path(args.out)
    cache_dir = None if args.no_cache else cache_dir_for(out_path)

    try:
        run_incremental(
            queries,
            out=out_path,
            location_code=args.location_code,
            language_code=args.language_code,
            max_organic=args.max_organic,
            crawl=not args.no_crawl,
            delay=args.delay,
            cache_dir=cache_dir,
        )
    except KeyboardInterrupt:
        print(
            "\nInterrupted -- partial raw progress is safe on disk "
            f"({raw_path_for(out_path)}). Re-run finalise manually if needed."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
