"""Shot freeze frames: where every visible player stands when a shot is taken.

StatsBomb stores them inside each shot as a list of players. This module
flattens that list into one row per player and per shot, ready for the
`shot_freeze_frames` table.
"""

import pandas as pd

COLUMNS = ["shot_id", "match_id", "player", "position", "teammate", "x", "y"]


def explode_freeze_frames(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """One row per player seen in each shot's freeze frame, for one match.

    The shooter is not in the list. Shots without a freeze frame (penalties,
    mostly) give no rows. The coordinates are in the same frame as the shot,
    with the shooting team attacking towards x = 120.
    """
    # A match where no shot has a freeze frame doesn't even have the column
    events = events.reindex(columns=events.columns.union(["shot_freeze_frame"]))
    shots = events.loc[events["type"].eq("Shot"), ["id", "shot_freeze_frame"]]
    # One row per player, then drop the shots that had nothing to explode
    shots = shots.explode("shot_freeze_frame").dropna(subset=["shot_freeze_frame"])
    if shots.empty:
        return pd.DataFrame(columns=COLUMNS)

    # Each player is a small dict: {"location": [x, y], "player": {...}, ...}
    players = pd.json_normalize(shots["shot_freeze_frame"].tolist())
    return pd.DataFrame({
        "shot_id": shots["id"].to_numpy(),
        "match_id": match_id,
        "player": players["player.name"].astype("string"),
        "position": players["position.name"].astype("string"),
        "teammate": players["teammate"].eq(True),
        "x": pd.to_numeric(players["location"].str[0]),
        "y": pd.to_numeric(players["location"].str[1]),
    })