"""Tests for the xG model code, on small made-up DataFrames."""

import math

import numpy as np
import pandas as pd
import pytest

from football_ai.xg import (
    build_boosting_model,
    build_logistic_model,
    coefficients,
    evaluate,
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


def test_boosting_gives_the_same_xg_when_trained_twice():
    shots = make_model_shots()
    first = predict_xg(train(build_boosting_model(), shots), shots)
    second = predict_xg(train(build_boosting_model(), shots), shots)

    assert first.equals(second)


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