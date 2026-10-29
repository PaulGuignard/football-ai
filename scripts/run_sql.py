"""Run the queries of a .sql file on the football database and print the results.

Each query must end with a semicolon. The comment lines just above a query
are printed as its title, so the output reads like a small report.

Usage:
    uv run python scripts/run_sql.py sql/explore/01_shots_overview.sql
"""

import sys
from pathlib import Path

import duckdb

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "football.duckdb"


def split_queries(text: str) -> list[tuple[str, str]]:
    """Split a .sql file into (title, query) pairs."""
    pairs = []
    for block in text.split(";"):
        lines = [line for line in block.strip().splitlines() if line.strip()]
        title = "\n".join(line for line in lines if line.lstrip().startswith("--"))
        query = "\n".join(line for line in lines if not line.lstrip().startswith("--"))
        if query:
            pairs.append((title, query))
    return pairs


def main() -> None:
    sql_file = Path(sys.argv[1])
    # read_only: exploring should never modify the data
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        for title, query in split_queries(sql_file.read_text()):
            print(title or query)
            con.sql(query).show(max_rows=50)
            print()


if __name__ == "__main__":
    main()