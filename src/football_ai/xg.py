"""Expected goals (xG) model: from the shot_features table to a trained model.

Loads the shots, splits them into a training set and a test set, and builds
the first model: a logistic regression.
"""

import duckdb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# The model inputs, grouped by how they need to be prepared
NUMERIC_FEATURES = ["distance", "angle"]
FLAG_FEATURES = ["is_header", "is_free_kick", "is_first_time", "is_one_on_one", "is_under_pressure"]
CATEGORICAL_FEATURES = ["play_pattern"]
FEATURES = NUMERIC_FEATURES + FLAG_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_goal"


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


def build_logistic_model() -> Pipeline:
    """A logistic regression with the preparation steps it needs, in one object.

    Distance and angle are standardised, the yes/no flags go through as 0/1 and
    play_pattern gets one column per value (one-hot). Being in the same
    pipeline as the model, these steps learn their settings (means, list of
    values) from the training set only.
    """
    preparation = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("flags", "passthrough", FLAG_FEATURES),
        # A value never seen in training just gets zeros instead of an error
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preparation", preparation),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def train(model: Pipeline, shots: pd.DataFrame) -> Pipeline:
    """Fit the model on these shots and return it."""
    return model.fit(shots[FEATURES], shots[TARGET])


def predict_xg(model: Pipeline, shots: pd.DataFrame) -> pd.Series:
    """The probability of a goal for each shot, i.e. its xG."""
    probabilities = model.predict_proba(shots[FEATURES])[:, 1]
    return pd.Series(probabilities, index=shots.index, name="xg")