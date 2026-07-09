#!/usr/bin/env python3
"""Generate the synthetic sample dataset committed to data/sample/.

Usage:
    python scripts/generate_sample_data.py [--n-queries 400] [--seed 42]
"""

from __future__ import annotations

import argparse

from aio_gap_miner import config
from aio_gap_miner.data import generate_synthetic_dataset, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-queries", type=int, default=400)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--out", type=str, default=str(config.SAMPLE_DATASET))
    args = parser.parse_args()

    df = generate_synthetic_dataset(n_queries=args.n_queries, seed=args.seed)
    path = save_dataset(df, args.out)

    pos = int(df["cited"].sum())
    print(
        f"Wrote {len(df):,} (query, URL) rows across {df['query_id'].nunique():,} queries to {path}"
    )
    print(f"Positives (cited): {pos:,}  ({pos / len(df):.1%} positive rate)")


if __name__ == "__main__":
    main()
