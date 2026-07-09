#!/usr/bin/env python3
"""Standalone ETL step: load the (query, URL) dataset into SQLite via SQLAlchemy.

    python scripts/build_database.py                 # from the synthetic sample
    python scripts/build_database.py --data data/raw/my_data.csv
"""

from __future__ import annotations

import argparse

from aio_gap_miner.data import load_dataset
from aio_gap_miner.database import (
    QUERY_CITATION_RATE_BY_RANK_BUCKET,
    build_database,
    read_sql,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()

    df = load_dataset(args.data)
    engine = build_database(df)

    # Demonstrate a query against the freshly built table.
    print("\nCitation rate by rank bucket:")
    print(read_sql(QUERY_CITATION_RATE_BY_RANK_BUCKET, engine).to_string(index=False))


if __name__ == "__main__":
    main()
