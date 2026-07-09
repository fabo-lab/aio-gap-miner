"""SQL / ETL layer for the AIO Gap-Miner (SQLAlchemy + SQLite).

A small but real ETL step: the flat (query, URL) CSV is loaded into a local
SQLite database via SQLAlchemy, and the pipeline reads its working set back out
with SQL. This mirrors how the project would sit on a warehouse in production
(swap the SQLite URL for Postgres/BigQuery and nothing else changes) and keeps
all feature logic queryable.

    from aio_gap_miner.database import build_database, read_sql
    engine = build_database(df)                 # CSV/DataFrame -> SQLite
    top = read_sql("SELECT * FROM candidates WHERE cited = 1 LIMIT 5", engine)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from . import config

TABLE_NAME: str = "candidates"
DB_PATH: Path = config.DATA_DIR / "aio_gap_miner.db"


def get_engine(db_path: str | Path | None = None) -> Engine:
    """Create (or connect to) the SQLite database engine."""
    db_path = Path(db_path) if db_path is not None else DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")


def build_database(
    df: pd.DataFrame,
    db_path: str | Path | None = None,
    table: str = TABLE_NAME,
) -> Engine:
    """ETL: write a (query, URL) DataFrame into a SQLite table (replace if exists)."""
    engine = get_engine(db_path)
    df.to_sql(table, engine, if_exists="replace", index=False)
    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    print(f"ETL: loaded {n:,} rows into '{table}' at {db_path or DB_PATH}")
    return engine


def read_sql(query: str, engine: Engine) -> pd.DataFrame:
    """Run a SQL query and return the result as a DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def load_candidates(engine: Engine, table: str = TABLE_NAME) -> pd.DataFrame:
    """Read the full candidates table back out via SQL."""
    return read_sql(f"SELECT * FROM {table}", engine)


# A couple of illustrative analytical SQL queries used in the notebook. Keeping
# them here documents the "SQL" competency and keeps them reusable.
QUERY_CITATION_RATE_BY_CONTENT_TYPE = f"""
    SELECT content_type,
           COUNT(*)                               AS candidates,
           SUM(cited)                             AS citations,
           ROUND(AVG(cited) * 100, 1)             AS citation_rate_pct
    FROM {TABLE_NAME}
    GROUP BY content_type
    ORDER BY citation_rate_pct DESC
"""

QUERY_CITATION_RATE_BY_RANK_BUCKET = f"""
    SELECT CASE
               WHEN organic_rank <= 3  THEN '1 top 3'
               WHEN organic_rank <= 10 THEN '2 rank 4-10'
               ELSE                         '3 rank 11+'
           END                                    AS rank_bucket,
           COUNT(*)                               AS candidates,
           SUM(cited)                             AS citations,
           ROUND(AVG(cited) * 100, 1)             AS citation_rate_pct
    FROM {TABLE_NAME}
    GROUP BY rank_bucket
    ORDER BY rank_bucket
"""
