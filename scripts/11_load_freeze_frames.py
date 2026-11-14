"""Load the shot freeze frames of every match already in the database.

Downloads the events of each match again, keeps only the freeze frames and
stores them in the `shot_freeze_frames` table (see sql/schema.sql). Run
04_load_events.py first: this script only goes through the matches it loaded.

Matches already done are listed in `freeze_frame_matches` and skipped, so the
script can be stopped and run again without duplicates. A full run takes
about 25 minutes, almost all of it spent downloading.

Usage:
    uv run python scripts/11_load_freeze_frames.py
"""

import time
import warnings
from pathlib import Path

import duckdb
from statsbombpy import sb

from football_ai.freeze_frames import explode_freeze_frames

# statsbombpy warns on every call when using open data, it's just noise
warnings.filterwarnings("ignore", message="credentials were not supplied")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "football.duckdb"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def load_match(con: duckdb.DuckDBPyConnection, match_id: int) -> int:
    """Store the freeze frames of one match and mark it as done. Returns the row count.

    Two inserts that must succeed or fail together: the players, and the line
    saying the match is done. A transaction makes sure we never keep one
    without the other.
    """
    frames = explode_freeze_frames(sb.events(match_id=match_id), match_id)
    con.register("frames", frames)
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute("INSERT INTO shot_freeze_frames BY NAME SELECT * FROM frames")
        con.execute("INSERT INTO freeze_frame_matches VALUES (?)", [match_id])
        con.execute("COMMIT")
    except Exception:
        # Stopped halfway: undo the players too, the match will be redone
        con.execute("ROLLBACK")
        raise
    return len(frames)


def main() -> None:
    """Load the freeze frames of every match not done yet, then print a summary."""
    start = time.time()
    with duckdb.connect(str(DB_PATH)) as con:
        con.execute(SCHEMA_PATH.read_text())
        to_load = [row[0] for row in con.execute("""
            SELECT match_id FROM matches
            EXCEPT
            SELECT match_id FROM freeze_frame_matches
            ORDER BY match_id
        """).fetchall()]
        print(f"{len(to_load)} matches to load")

        for i, match_id in enumerate(to_load, start=1):
            load_match(con, match_id)
            if i % 50 == 0 or i == len(to_load):
                print(f"  {i}/{len(to_load)} matches loaded")

        n_shots, n_players = con.execute(
            "SELECT COUNT(DISTINCT shot_id), COUNT(*) FROM shot_freeze_frames"
        ).fetchone()
    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f} s: {n_shots} shots, {n_players} players")


if __name__ == "__main__":
    main()