"""Load the 2015/16 events of four leagues into DuckDB.

Downloads every match of Premier League, La Liga, Serie A and Ligue 1
2015/16 from StatsBomb open data, cleans the events and stores them in
data/football.duckdb (tables `matches` and `events`, see sql/schema.sql).

Matches already in the database are skipped, so the script can be stopped
and run again without creating duplicates. A full run takes about 25 minutes.

Usage:
    uv run python scripts/04_load_events.py
"""

import time
import warnings
from pathlib import Path

import duckdb
import pandas as pd
from statsbombpy import sb

# statsbombpy warns on every call when using open data, it's just noise
warnings.filterwarnings("ignore", message="credentials were not supplied")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "football.duckdb"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"

# Complete seasons only (see 02_count_matchs.py), the others are partial
COMPETITIONS = [
    ("Premier League", 2, 27),
    ("La Liga", 11, 27),
    ("Serie A", 12, 27),
    ("Ligue 1", 7, 27),
]

# StatsBomb only fills these when True, otherwise the cell is just empty
BOOLEAN_COLUMNS = [
    "under_pressure", "pass_shot_assist", "pass_goal_assist",
    "shot_first_time", "shot_one_on_one",
]
TEXT_COLUMNS = [
    "event_id", "possession_team", "play_pattern", "team", "player", "position",
    "type", "pass_height", "pass_body_part", "pass_type", "pass_outcome",
    "dribble_outcome", "shot_outcome", "shot_type", "shot_body_part",
    "shot_technique", "shot_key_pass_id",
]
NUMBER_COLUMNS = ["duration", "pass_length", "shot_statsbomb_xg"]
# [x, y] lists split into two columns, much easier to query in SQL
LOCATION_COLUMNS = {
    "location": ("x", "y"),
    "pass_end_location": ("pass_end_x", "pass_end_y"),
    "carry_end_location": ("carry_end_x", "carry_end_y"),
    "shot_end_location": ("shot_end_x", "shot_end_y"),
}


def clean_events(raw: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Turn the raw events of one match into rows matching the `events` table.

    Keeps only the columns of the schema, gives each one its SQL type and
    splits the [x, y] locations into separate columns.
    """
    df = raw.rename(columns={"id": "event_id", "index": "event_index"})
    # Some columns don't exist when the event never happens in a match
    # (no one-on-one shot for instance), so we create them empty
    needed = BOOLEAN_COLUMNS + TEXT_COLUMNS + NUMBER_COLUMNS + list(LOCATION_COLUMNS)
    df = df.reindex(columns=df.columns.union(needed))

    clean = pd.DataFrame({"match_id": match_id}, index=df.index)
    for col in ["event_index", "period", "minute", "second", "possession"]:
        clean[col] = df[col].astype("int64")
    for col in TEXT_COLUMNS:
        clean[col] = df[col].astype("string")  # missing text -> NULL, not "nan"
    for col in NUMBER_COLUMNS:
        clean[col] = pd.to_numeric(df[col])
    for col in BOOLEAN_COLUMNS:
        clean[col] = df[col].eq(True)
    for source, (x_col, y_col) in LOCATION_COLUMNS.items():
        clean[x_col] = pd.to_numeric(df[source].str[0])
        clean[y_col] = pd.to_numeric(df[source].str[1])
    return clean


def load_competition(con, name: str, competition_id: int, season_id: int) -> None:
    """Load the matches and events of one competition-season.

    Matches are inserted first (duplicates ignored), then the events of every
    match not yet in the `events` table, one match at a time.
    """
    # Matches first, the PRIMARY KEY makes re-runs safe
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    matches_clean = pd.DataFrame({
        "match_id": matches["match_id"],
        "competition": name,
        "season": matches["season"],
        "match_date": pd.to_datetime(matches["match_date"]).dt.date,
        "home_team": matches["home_team"],
        "away_team": matches["away_team"],
        "home_score": matches["home_score"],
        "away_score": matches["away_score"],
    })
    con.register("matches_clean", matches_clean)
    con.execute("INSERT OR IGNORE INTO matches BY NAME SELECT * FROM matches_clean")

    # Events: only download the matches we don't have yet
    already_loaded = {
        row[0] for row in con.execute("SELECT DISTINCT match_id FROM events").fetchall()
    }
    to_load = [m for m in matches["match_id"] if m not in already_loaded]
    print(f"{name}: {len(matches)} matches, {len(to_load)} to load")

    for i, match_id in enumerate(to_load, start=1):
        events_clean = clean_events(sb.events(match_id=match_id), match_id)
        con.register("events_clean", events_clean)
        # One INSERT per match, so a match is never half loaded
        con.execute("INSERT INTO events BY NAME SELECT * FROM events_clean")
        if i % 25 == 0 or i == len(to_load):
            print(f"  {i}/{len(to_load)} matches loaded")


def main() -> None:
    """Create the tables if needed, load each competition and print a summary."""
    DB_PATH.parent.mkdir(exist_ok=True)
    start = time.time()
    with duckdb.connect(str(DB_PATH)) as con:
        con.execute(SCHEMA_PATH.read_text())
        for name, competition_id, season_id in COMPETITIONS:
            load_competition(con, name, competition_id, season_id)

        n_matches, n_events = con.execute(
            "SELECT COUNT(DISTINCT match_id), COUNT(*) FROM events"
        ).fetchone()
    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f} s: {n_matches} matches, {n_events} events")


if __name__ == "__main__":
    main()