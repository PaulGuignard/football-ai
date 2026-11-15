"""Tests for the Deep Sets xG model, on hand-made shots and players."""

import numpy as np
import pandas as pd
import pytest

from football_ai.deep_sets import MAX_PLAYERS, PLAYER_FEATURES, player_arrays


def make_players(rows):
    """Players as load_players returns them, every shot taken from the penalty spot."""
    players = pd.DataFrame(rows, columns=["shot_id", "x", "y", "teammate", "position"])
    players["shot_x"], players["shot_y"] = 108.0, 40.0
    return players


def test_each_shot_gets_its_players_and_a_matching_mask():
    shots = pd.DataFrame({"shot_id": ["a", "b"]})
    players = make_players([
        ("a", 114.0, 40.0, False, "Center Back"),
        ("a", 119.0, 40.0, False, "Goalkeeper"),
        ("b", 110.0, 30.0, True, "Left Wing"),
    ])
    positions, mask = player_arrays(shots, players)

    assert positions.shape == (2, MAX_PLAYERS, len(PLAYER_FEATURES))
    assert mask[0].tolist() == [1, 1] + [0] * (MAX_PLAYERS - 2)
    assert mask[1].tolist() == [1] + [0] * (MAX_PLAYERS - 1)
    # Padding stays at zero
    assert (positions[1, 1:] == 0).all()


def test_player_features_are_relative_to_the_ball_and_the_goal():
    shots = pd.DataFrame({"shot_id": ["a"]})
    players = make_players([("a", 114.0, 38.0, False, "Goalkeeper")])
    positions, _ = player_arrays(shots, players)

    # 6 yards ahead of the ball and 2 to its side, 6 yards off the goal line
    assert positions[0, 0].tolist() == pytest.approx([0.6, -0.2, 0.6, 0.2, 0.0, 1.0])


def test_rows_follow_the_order_of_the_shots_not_of_the_players():
    shots = pd.DataFrame({"shot_id": ["b", "a"]})
    players = make_players([
        ("a", 114.0, 40.0, False, "Center Back"),
        ("b", 110.0, 40.0, True, "Left Wing"),
    ])
    positions, _ = player_arrays(shots, players)

    # Row 0 is shot "b", the teammate
    assert positions[0, 0, PLAYER_FEATURES.index("teammate")] == 1.0
    assert positions[1, 0, PLAYER_FEATURES.index("teammate")] == 0.0


def test_a_shot_with_nobody_in_the_picture_is_all_padding():
    shots = pd.DataFrame({"shot_id": ["a", "empty"]})
    players = make_players([("a", 114.0, 40.0, False, "Center Back")])
    positions, mask = player_arrays(shots, players)

    assert mask[1].sum() == 0
    assert (positions[1] == 0).all()


def test_too_many_players_is_refused():
    shots = pd.DataFrame({"shot_id": ["a"]})
    players = make_players([("a", 100.0 + i % 20, 40.0, False, "Center Back")
                            for i in range(MAX_PLAYERS + 1)])

    with pytest.raises(ValueError):
        player_arrays(shots, players)


def test_players_of_shots_not_asked_for_are_ignored():
    shots = pd.DataFrame({"shot_id": ["a"]})
    players = make_players([
        ("a", 114.0, 40.0, False, "Center Back"),
        ("penalty", 119.0, 40.0, False, "Goalkeeper"),
    ])
    _, mask = player_arrays(shots, players)

    assert mask.sum() == 1
    assert np.isfinite(mask).all()