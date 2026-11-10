"""Expected goals (xG) model: from the shot_features table to a trained model.

This first part loads the shots and splits them into a training set and a
test set. The model itself comes next.
"""

import duckdb
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def load_shots(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Read the shot_features table into a DataFrame, one row per shot."""
    return con.sql("SELECT * FROM shot_features").df()


def split_by_match(
    shots: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split shots into a training set and a test set, keeping each match whole.

    Shots from the same match share teams, keepers and game state, so a match
    must never have shots on both sides. test_size is the share of matches,
    not of shots, set aside for the test.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(shots, groups=shots["match_id"]))
    return shots.iloc[train_idx], shots.iloc[test_idx]