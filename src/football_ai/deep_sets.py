"""xG from the raw positions of the players, with a Deep Sets neural network.

The players seen in a freeze frame form a set: their number changes from one
shot to the next and they come in no particular order. This module first
turns each shot's players into fixed-size arrays the network can read, with a
mask saying which slots hold a real player.
"""

import duckdb
import numpy as np
import pandas as pd

# A freeze frame never shows more than the 21 players other than the shooter
MAX_PLAYERS = 21
PLAYER_FEATURES = [
    "dx_shooter",  # where the player is, seen from the ball
    "dy_shooter",
    "dx_goal",  # and seen from the centre of the goal
    "dy_goal",
    "teammate",
    "keeper",
]
# Yards divided by 10, so that positions stay around -1 to 1 like the rest
SCALE = 10.0


def load_players(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Every player of every freeze frame, next to the location of the shot."""
    return con.sql("""
        SELECT f.shot_id, f.x, f.y, f.teammate, f.position, e.x AS shot_x, e.y AS shot_y
        FROM shot_freeze_frames AS f
        JOIN events AS e ON e.event_id = f.shot_id
    """).df()


def player_arrays(shots: pd.DataFrame, players: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The players of each shot as fixed-size arrays, in the order of `shots`.

    Returns `positions`, shaped (shots, MAX_PLAYERS, features), and `mask`,
    shaped (shots, MAX_PLAYERS), with 1 where a slot holds a real player and
    0 where it is empty padding. A shot with nobody in the picture gets an
    all-zero row and an all-zero mask.
    """
    row_of_shot = pd.Series(np.arange(len(shots)), index=shots["shot_id"].to_numpy())
    players = players[players["shot_id"].isin(row_of_shot.index)].copy()
    players["row"] = players["shot_id"].map(row_of_shot)
    # Which slot each player takes within its shot: 0, 1, 2...
    players["slot"] = players.groupby("row").cumcount()
    if (players["slot"] >= MAX_PLAYERS).any():
        raise ValueError(f"A freeze frame has more than {MAX_PLAYERS} players")

    features = np.column_stack([
        (players["x"] - players["shot_x"]) / SCALE,
        (players["y"] - players["shot_y"]) / SCALE,
        (120 - players["x"]) / SCALE,
        (40 - players["y"]) / SCALE,
        players["teammate"].astype(float),
        players["position"].eq("Goalkeeper").astype(float),
    ]).astype(np.float32)

    positions = np.zeros((len(shots), MAX_PLAYERS, len(PLAYER_FEATURES)), dtype=np.float32)
    mask = np.zeros((len(shots), MAX_PLAYERS), dtype=np.float32)
    rows, slots = players["row"].to_numpy(), players["slot"].to_numpy()
    positions[rows, slots] = features
    mask[rows, slots] = 1.0
    return positions, mask