"""Expected goals (xG) model: from the feature tables to a trained model.

Loads the shots, splits them into a training set and a test set, builds the
models (a logistic regression, then gradient boosting), scores their
predictions, cross-validates settings and shows what the logistic regression
learnt. Two sets of inputs: "v1" is what the event data says about the shot,
"v2" adds what the freeze frame shows (defenders in the way, the keeper).
"""

from collections.abc import Callable

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# The model inputs, grouped by how they need to be prepared
NUMERIC_FEATURES = ["distance", "angle"]
FLAG_FEATURES = ["is_header", "is_free_kick", "is_first_time", "is_one_on_one", "is_under_pressure"]
CATEGORICAL_FEATURES = ["play_pattern"]
FEATURES = NUMERIC_FEATURES + FLAG_FEATURES + CATEGORICAL_FEATURES
# From the freeze frames, all numbers. Some are missing when nobody, or no
# single keeper, is in the picture
FREEZE_FEATURES = ["defenders_in_cone", "nearest_defender", "keeper_to_goal", "keeper_in_cone"]
FEATURE_SETS = {"v1": FEATURES, "v2": FEATURES + FREEZE_FEATURES}
TARGET = "is_goal"


def load_shots(con: duckdb.DuckDBPyConnection, freeze_frames: bool = False) -> pd.DataFrame:
    """Read the shots into a DataFrame, one row per shot.

    With freeze_frames=True, the columns of shot_freeze_features are added
    (that table has to be built first, see football_ai.features).
    """
    if not freeze_frames:
        return con.sql("SELECT * FROM shot_features").df()
    shots = con.sql("""
        SELECT *
        FROM shot_features
        LEFT JOIN shot_freeze_features USING (shot_id)
    """).df()
    # DuckDB hands back integer columns with gaps as pandas' nullable integers
    # (pd.NA), which scikit-learn refuses. Plain floats with NaN instead
    shots[FREEZE_FEATURES] = shots[FREEZE_FEATURES].astype("float64")
    return shots


def numeric_inputs(feature_set: str) -> list[str]:
    """The numeric inputs of a feature set: distance and angle, plus the freeze ones in v2."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}, expected one of {list(FEATURE_SETS)}")
    return NUMERIC_FEATURES + (FREEZE_FEATURES if feature_set == "v2" else [])


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


def build_logistic_model(feature_set: str = "v1") -> Pipeline:
    """A logistic regression with the preparation steps it needs, in one object.

    Numbers are standardised, the yes/no flags go through as 0/1 and
    play_pattern gets one column per value (one-hot). A logistic regression
    can't take a missing value, so a missing number is replaced by the median
    of the training shots. Being in the same pipeline as the model, these steps
    learn their settings (medians, means, list of values) from the training
    set only.
    """
    preparation = ColumnTransformer([
        # In v1 nothing is missing and the imputer changes nothing
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
         numeric_inputs(feature_set)),
        ("flags", "passthrough", FLAG_FEATURES),
        # A value never seen in training just gets zeros instead of an error
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preparation", preparation),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def build_boosting_model(feature_set: str = "v1", seed: int = 42, **settings) -> Pipeline:
    """Gradient boosting: many small trees, each one fixing what the others got wrong.

    Without settings, scikit-learn's defaults. Any setting of the model can be
    passed by name, e.g. build_boosting_model(learning_rate=0.05). Early
    stopping is on by default above 10,000 shots: 10% of the training shots
    are held back and trees stop being added once they no longer help there.
    The seed fixes which shots those are. Missing values need no filling in:
    at each question, the trees learn which way to send them.
    """
    preparation = ColumnTransformer([
        # Trees only ask "is distance below 9?", so the scale doesn't matter
        ("numeric", "passthrough", numeric_inputs(feature_set)),
        ("flags", "passthrough", FLAG_FEATURES),
        # Dense output: this model doesn't accept sparse matrices
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preparation", preparation),
        ("model", HistGradientBoostingClassifier(random_state=seed, **settings)),
    ])


def model_inputs(model: Pipeline) -> list[str]:
    """The columns a model reads, in order, taken from its preparation step."""
    preparation = model.named_steps["preparation"]
    return [column for _, _, columns in preparation.transformers for column in columns]


def train(model: Pipeline, shots: pd.DataFrame) -> Pipeline:
    """Fit the model on these shots and return it."""
    return model.fit(shots[model_inputs(model)], shots[TARGET])


def predict_xg(model: Pipeline, shots: pd.DataFrame) -> pd.Series:
    """The probability of a goal for each shot, i.e. its xG."""
    probabilities = model.predict_proba(shots[model_inputs(model)])[:, 1]
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