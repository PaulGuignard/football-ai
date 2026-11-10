"""Build the shot_features table that the xG model learns from.

The SQL itself lives in sql/features/shot_features.sql. This module runs it on
a DuckDB connection, so the command line and the tests use the exact same code.

Usage:
    uv run python -m football_ai.features
"""

from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "football.duckdb"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"
FEATURES_SQL_PATH = PROJECT_ROOT / "sql" / "features" / "shot_features.sql"


def build_shot_features(con: duckdb.DuckDBPyConnection) -> int:
    """Create (or replace) the shot_features table and return its number of rows."""
    con.execute(FEATURES_SQL_PATH.read_text())
    return con.execute("SELECT COUNT(*) FROM shot_features").fetchone()[0]


def main() -> None:
    """Build the table in the project database and print how many shots it holds."""
    with duckdb.connect(str(DB_PATH)) as con:
        n_shots = build_shot_features(con)
    print(f"shot_features: {n_shots:,} shots")


if __name__ == "__main__":
    main()