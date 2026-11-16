"""Check on every real shot that the Python inputs match the SQL ones.

The model is trained on inputs computed in SQL and the API computes them in
Python. This recomputes all of them in Python for the 37,481 shots and
compares. Any difference is listed; none is expected.

Usage:
    uv run python scripts/14_check_shot_inputs.py
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from football_ai.shot_inputs import shot_row
from football_ai.xg import FREEZE_FEATURES, load_shots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "football.duckdb"
COLUMNS = ["distance", "angle", *FREEZE_FEATURES]


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        sql = load_shots(con, freeze_frames=True).set_index("shot_id")
        location = con.sql(
            "SELECT event_id AS shot_id, x, y FROM events WHERE type = 'Shot'"
        ).df().set_index("shot_id")
        players = con.sql("""
            SELECT shot_id, x, y, teammate, position = 'Goalkeeper' AS keeper
            FROM shot_freeze_frames
        """).df()

    players_by_shot = {
        shot_id: group[["x", "y", "teammate", "keeper"]].to_dict("records")
        for shot_id, group in players.groupby("shot_id")
    }
    python = pd.concat([
        shot_row(location.loc[shot_id, "x"], location.loc[shot_id, "y"],
                 players_by_shot.get(shot_id, []))
        for shot_id in sql.index
    ]).set_index(sql.index)

    print(f"{len(sql):,} shots checked")
    for column in COLUMNS:
        a, b = sql[column].to_numpy(dtype=float), python[column].to_numpy(dtype=float)
        same_gaps = np.array_equal(np.isnan(a), np.isnan(b))
        both = ~np.isnan(a) & ~np.isnan(b)
        largest_gap = np.abs(a[both] - b[both]).max()
        print(f"  {column:<18} missing in the same places: {same_gaps}, "
              f"largest difference: {largest_gap:.1e}")


if __name__ == "__main__":
    main()