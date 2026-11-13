"""Expected goals (xG) model: from the shot_features table to a trained model.

Loads the shots, splits them into a training set and a test set, builds the
models (a logistic regression, then gradient boosting), scores their
predictions, cross-validates settings and shows what the logistic regression
learnt.
"""

from collections.abc import Callable

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
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


def build_boosting_model(seed: int = 42, **settings) -> Pipeline:
    """Gradient boosting: many small trees, each one fixing what the others got wrong.

    Without settings, scikit-learn's defaults. Any setting of the model can be
    passed by name, e.g. build_boosting_model(learning_rate=0.05). Early
    stopping is on by default above 10,000 shots: 10% of the training shots
    are held back and trees stop being added once they no longer help there.
    The seed fixes which shots those are.
    """
    preparation = ColumnTransformer([
        # Trees only ask "is distance below 9?", so the scale doesn't matter
        ("numeric", "passthrough", NUMERIC_FEATURES),
        ("flags", "passthrough", FLAG_FEATURES),
        # Dense output: this model doesn't accept sparse matrices
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preparation", preparation),
        ("model", HistGradientBoostingClassifier(random_state=seed, **settings)),
    ])


def train(model: Pipeline, shots: pd.DataFrame) -> Pipeline:
    """Fit the model on these shots and return it."""
    return model.fit(shots[FEATURES], shots[TARGET])


def predict_xg(model: Pipeline, shots: pd.DataFrame) -> pd.Series:
    """The probability of a goal for each shot, i.e. its xG."""
    probabilities = model.predict_proba(shots[FEATURES])[:, 1]
    return pd.Series(probabilities, index=shots.index, name="xg")


def evaluate(is_goal: pd.Series, xg: pd.Series) -> dict[str, float]:
    """Score xG predictions against what really happened.

    log_loss and brier: lower is better. auc: 0.5 is a coin toss, 1 is
    perfect ranking. expected_goals should be close to goals.
    """
    return {
        "log_loss": log_loss(is_goal, xg, labels=[0, 1]),
        "brier": brier_score_loss(is_goal, xg),
        "auc": roc_auc_score(is_goal, xg),
        "goals": float(is_goal.sum()),
        "expected_goals": float(xg.sum()),
    }


def cross_validate_by_match(
    build_model: Callable[[], Pipeline], shots: pd.DataFrame, n_splits: int = 5
) -> np.ndarray:
    """Log loss of a model on each of n_splits folds of whole matches.

    The matches are put into n_splits groups. Each group in turn is held out:
    the model learns on the others and is scored on it. Meant for the training
    set only, so that settings are chosen without ever looking at the test set.
    The folds don't depend on chance, so every setting is scored on exactly
    the same ones and can be compared fold by fold.
    """
    losses = []
    folds = GroupKFold(n_splits=n_splits).split(shots, groups=shots["match_id"])
    for learn_idx, check_idx in folds:
        learn, check = shots.iloc[learn_idx], shots.iloc[check_idx]
        model = train(build_model(), learn)
        losses.append(log_loss(check[TARGET], predict_xg(model, check), labels=[0, 1]))
    return np.array(losses)


def coefficients(model: Pipeline) -> pd.DataFrame:
    """What the logistic regression learnt: one row per model input.

    odds_ratio = exp(coefficient): by how much the odds of scoring are
    multiplied when the input goes up by one (one standard deviation for
    distance and angle, "yes" instead of "no" for the flags), everything else
    being equal. Above 1 helps, below 1 hurts.
    """
    names = model.named_steps["preparation"].get_feature_names_out()
    coefs = model.named_steps["model"].coef_[0]
    table = pd.DataFrame({"input": names, "coefficient": coefs})
    # "numeric__distance" -> "distance", "categorical__play_pattern_From Counter" -> "From Counter"
    table["input"] = (
        table["input"].str.split("__").str[1].str.replace("play_pattern_", "", regex=False)
    )
    table["odds_ratio"] = np.exp(table["coefficient"])
    return table.sort_values("coefficient").reset_index(drop=True)