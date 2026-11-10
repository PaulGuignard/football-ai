"""Tests for the xG model code, on small made-up DataFrames."""

import pandas as pd

from football_ai.xg import split_by_match


def make_shots(n_matches: int = 10, shots_per_match: int = 3) -> pd.DataFrame:
    """A fake shot table: a few shots per match, nothing else needed here."""
    return pd.DataFrame({
        "match_id": [m for m in range(n_matches) for _ in range(shots_per_match)],
        "is_goal": [0, 0, 1] * n_matches,
    })


def test_no_match_ends_up_on_both_sides():
    train, test = split_by_match(make_shots())

    assert set(train["match_id"]).isdisjoint(test["match_id"])


def test_every_shot_is_kept_exactly_once():
    shots = make_shots()
    train, test = split_by_match(shots)

    assert len(train) + len(test) == len(shots)
    assert sorted(train.index.tolist() + test.index.tolist()) == shots.index.tolist()


def test_about_20_percent_of_matches_go_to_test():
    _, test = split_by_match(make_shots(n_matches=10))

    assert test["match_id"].nunique() == 2


def test_same_seed_gives_same_split():
    shots = make_shots()
    first_test = split_by_match(shots, seed=7)[1]
    second_test = split_by_match(shots, seed=7)[1]

    assert first_test.index.tolist() == second_test.index.tolist()