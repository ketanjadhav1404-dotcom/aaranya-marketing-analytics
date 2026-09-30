"""
db.py - one place for the database connection.

Every notebook, script and the copilot import from here, so you only
configure the connection once, in a file called `.env` next to this file:

    DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/aaranya

Usage in a notebook:
    import sys; sys.path.append("..")
    from db import engine, q
    df = q("SELECT * FROM mart.customer LIMIT 5")
"""
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:  # python-dotenv not installed: fall back to real env vars
    pass

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/aaranya"
)
engine = create_engine(DATABASE_URL)


def q(sql: str, **params) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame. Use :name placeholders for params."""
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def write(df: pd.DataFrame, table: str, schema: str = "mart") -> None:
    """Replace a table in the warehouse with a DataFrame (used for model outputs)."""
    df.to_sql(table, engine, schema=schema, if_exists="replace", index=False)


if __name__ == "__main__":
    print(q("SELECT current_database() AS db, now() AS connected_at"))
