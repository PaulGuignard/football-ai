"""Tests for the xG model code, on small made-up DataFrames."""

import math

import numpy as np
import pandas as pd
import pytest

from football_ai.xg import (
    FREEZE_FEATURES,
    build_boosting_model,
    build_logistic_model,
    coefficients,
    cross_validate_by_match,
    evaluate,
    model_inputs,
    predict_xg,
    split_by_match,
    train,
)

# Every model must pass the same basic checks
MODEL_BUILDERS = [build_logistic_model, build_boosting_model]


def make_shots(n_matches: int = 10, shots_per_match: int = 3) -> pd.DataFrame:
    """A fake shot table: a few shots per match, nothing else needed here."""
    return pd.DataFrame({
        "match_id": [m for m in range(n_matches) for _ in range(shots_per_match)],
        "is_goal": [0, 0, 1] * n_matches,
    })


def make_model_shots(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Fake shots with every model input, where closer shots score more often."""
    rng = np.random.default_rng(seed)
    distance = rng.uniform(2, 40, n)
    return pd.DataFrame({
        "match_id": np.arange(n) // 4,
        "distance": distance,
        "angle": rng.uniform(5, 60, n),
        "is_header": rng.random(n) < 0.2,
        "is_free_kick": rng.random(n) < 0.05,
        "is_first_time": rng.random(n) < 0.3,
        "is_one_on_one": rng.random(n) < 0.05,
        "is_under_pressure": rng.random(n) < 0.25,
        "play_pattern": rng.choice(["Regular Play", "From Counter", "From Corner"], n),
        "is_goal": (rng.random(n) < 0.6 * np.exp(-distance / 8)).astype(int),
    })


def make_v2_shots(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """The same fake shots plus freeze frame inputs, some of them missing."""
    rng = np.random.default_rng(seed)
    shots = make_model_shots(n, seed)
    shots["defenders_in_cone"] = rng.integers(0, 4, n)
    shots["nearest_defender"] = rng.uniform(0, 8, n)
    shots["keeper_to_goal"] = rng.uniform(0, 10, n)
    shots["keeper_in_cone"] = (rng.random(n) < 0.9).astype(float)
    # Every tenth shot has no keeper in the picture, every 20th nobody at all
    shots.loc[shots.index % 10 == 0, ["keeper_to_goal", "keeper_in_cone"]] = np.nan
    shots.loc[shots.index % 20 == 0, "nearest_defender"] = np.nan
    return shots


def test_no_match_ends_up_on_both_sides():
    train_set, test_set = split_by_match(make_shots())

    assert set(train_set["match_id"]).isdisjoint(test_set["match_id"])


def test_every_shot_is_kept_exactly_once():
    shots = make_shots()
    train_set, test_set = split_by_match(shots)

    assert len(train_set) + len(test_set) == len(shots)
    assert sorted(train_set.index.tolist() + test_set.index.tolist()) == shots.index.tolist()


def test_about_20_percent_of_matches_go_to_test():
    _, test_set = split_by_match(make_shots(n_matches=10))

    assert test_set["match_id"].nunique() == 2


def test_same_seed_gives_same_split():
    shots = make_shots()
    first_test = split_by_match(shots, seed=7)[1]
    second_test = split_by_match(shots, seed=7)[1]

    assert first_test.index.tolist() == second_test.index.tolist()


@pytest.mark.parametrize("build_model", MODEL_BUILDERS)
def test_xg_is_a_probability_for_every_shot(build_model):
    shots = make_model_shots()
    model = train(build_model(), shots)
    xg = predict_xg(model, shots)

    assert len(xg) == len(shots)
    assert xg.between(0, 1).all()


@pytest.mark.parametrize("build_model", MODEL_BUILDERS)
def test_model_learns_that_closer_shots_are_more_dangerous(build_model):
    shots = make_model_shots()
    model = train(build_model(), shots)

    close, far = shots.iloc[[0]].copy(), shots.iloc[[0]].copy()
    close["distance"], far["distance"] = 5.0, 35.0

    assert predict_xg(model, close).iloc[0] > predict_xg(model, far).iloc[0]


@pytest.mark.parametrize("build_model", MODEL_BUILDERS)
def test_unknown_play_pattern_does_not_break_predictions(build_model):
    shots = make_model_shots()
    model = train(build_model(), shots)

    new_shot = shots.iloc[[0]].copy()
    new_shot["play_pattern"] = "From Keeper"  # never seen in training

    assert predict_xg(model, new_shot).between(0, 1).all()


@pytest.mark.parametrize("build_model", MODEL_BUILDERS)
def test_v2_models_cope_with_missing_values(build_model):
    shots = make_v2_shots()
    model = train(build_model("v2"), shots)

    assert predict_xg(model, shots).between(0, 1).all()


@pytest.mark.parametrize("build_model", MODEL_BUILDERS)
def test_only_v2_models_read_the_freeze_frame_inputs(build_model):
    v1_inputs = model_inputs(build_model("v1"))
    v2_inputs = model_inputs(build_model("v2"))

    assert not set(FREEZE_FEATURES) & set(v1_inputs)
    assert set(FREEZE_FEATURES) <= set(v2_inputs)


def test_a_v1_model_ignores_extra_columns():
    # Same shots with or without the freeze columns: same v1 predictions
    v1_xg = predict_xg(train(build_logistic_model(), make_model_shots()), make_model_shots())
    v2_shots = make_v2_shots()
    xg_with_extra = predict_xg(train(build_logistic_model(), v2_shots), v2_shots)

    assert np.allclose(v1_xg, xg_with_extra)


def test_unknown_feature_set_is_refused():
    with pytest.raises(ValueError):
        build_logistic_model("v3")


def test_boosting_gives_the_same_xg_when_trained_twice():
    shots = make_model_shots()
    first = predict_xg(train(build_boosting_model(), shots), shots)
    second = predict_xg(train(build_boosting_model(), shots), shots)

    assert first.equals(second)


def test_boosting_settings_reach_the_model():
    model = build_boosting_model(learning_rate=0.05, max_leaf_nodes=7)

    assert model.named_steps["model"].learning_rate == 0.05
    assert model.named_steps["model"].max_leaf_nodes == 7


def test_cross_validation_gives_one_log_loss_per_fold():
    losses = cross_validate_by_match(build_logistic_model, make_model_shots(), n_splits=5)

    assert len(losses) == 5
    assert (losses > 0).all()


def test_cross_validation_folds_depend_on_the_matches_not_the_row_order():
    shots = make_model_shots()
    shuffled = shots.sample(frac=1, random_state=3)

    first = cross_validate_by_match(build_logistic_model, shots)
    second = cross_validate_by_match(build_logistic_model, shuffled)

    assert np.allclose(first, second)


def test_cross_validation_folds_are_the_same_every_time():
    shots = make_model_shots()
    first = cross_validate_by_match(build_logistic_model, shots)
    second = cross_validate_by_match(build_logistic_model, shots)

    assert (first == second).all()


def test_perfect_predictions_get_perfect_scores():
    is_goal = pd.Series([0, 1, 0, 1])
    scores = evaluate(is_goal, is_goal.astype(float))

    assert scores["brier"] == 0
    assert scores["auc"] == 1


def test_saying_50_percent_everywhere_gives_the_textbook_log_loss():
    is_goal = pd.Series([0, 1, 0, 1])
    scores = evaluate(is_goal, pd.Series([0.5] * 4))

    # -ln(0.5), the log loss of a coin toss
    assert scores["log_loss"] == pytest.approx(math.log(2))
    assert scores["auc"] == 0.5


def test_expected_goals_is_the_sum_of_xg():
    scores = evaluate(pd.Series([0, 1, 0]), pd.Series([0.1, 0.6, 0.2]))

    assert scores["goals"] == 1
    assert scores["expected_goals"] == pytest.approx(0.9)


@pytest.mark.parametrize("feature_set", ["v1", "v2"])
def test_a_monotonic_boosting_never_rises_with_distance(feature_set):
    shots = make_v2_shots()
    # A few far shots that went in: the kind of noise the constraint is for
    far = shots["distance"] > 35
    shots.loc[far[far].index[:6], "is_goal"] = 1
    model = train(build_boosting_model(feature_set, monotonic=True), shots)

    same_shot_further_out = pd.concat([shots.iloc[[1]]] * 40, ignore_index=True)
    same_shot_further_out["distance"] = np.linspace(2, 60, 40)
    xg = predict_xg(model, same_shot_further_out)

    assert (np.diff(xg) <= 1e-12).all()


def test_the_monotonic_option_reaches_the_model():
    plain = build_boosting_model("v2").named_steps["model"]
    constrained = build_boosting_model("v2", monotonic=True).named_steps["model"]

    assert plain.monotonic_cst is None
    assert constrained.monotonic_cst == {"numeric__distance": -1, "numeric__angle": 1}


def test_coefficients_have_one_row_per_input_and_readable_names():
    shots = make_model_shots()
    table = coefficients(train(build_logistic_model(), shots))

    # 2 numbers + 5 flags + 3 play patterns in the fake data
    assert len(table) == 10
    assert {"distance", "is_header", "From Counter"} <= set(table["input"])


def test_distance_lowers_the_odds_on_data_built_that_way():
    shots = make_model_shots()
    table = coefficients(train(build_logistic_model(), shots)).set_index("input")

    assert table.loc["distance", "odds_ratio"] < 1