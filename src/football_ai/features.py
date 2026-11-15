"""Build the tables that the xG models learn from.

shot_features holds one row per shot with what the event data says about it.
shot_freeze_features adds what the freeze frame shows: defenders in the way
and where the keeper stands. The SQL lives in sql/features/. This module runs
it on a DuckDB connection, so the command line and the tests use the exact
same code.

Usage:
    uv run python -m football_ai.features
"""

from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "football.duckdb"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"
FEATURES_SQL_PATH = PROJECT_ROOT / "sql" / "features" / "shot_features.sql"
FREEZE_FEATURES_SQL_PATH = PROJECT_ROOT / "sql" / "features" / "shot_freeze_features.sql"


def build_shot_features(con: duckdb.DuckDBPyConnection) -> int:
    """Create (or replace) the shot_features table and return its number of rows."""
    con.execute(FEATURES_SQL_PATH.read_text())
    return con.execute("SELECT COUNT(*) FROM shot_features").fetchone()[0]


def build_shot_freeze_features(con: duckdb.DuckDBPyConnection) -> int:
    """Create (or replace) the shot_freeze_features table and return its number of rows.

    Needs shot_features to exist: it holds the shots to describe.
    """
    con.execute(FREEZE_FEATURES_SQL_PATH.read_text())
    return con.execute("SELECT COUNT(*) FROM shot_freeze_features").fetchone()[0]


def main() -> None:
    """Build both tables in the project database and print their sizes."""
    with duckdb.connect(str(DB_PATH)) as con:
        n_shots = build_shot_features(con)
        n_frames = build_shot_freeze_features(con)
    print(f"shot_features: {n_shots:,} shots")
    print(f"shot_freeze_features: {n_frames:,} shots")


if __name__ == "__main__":
    main()