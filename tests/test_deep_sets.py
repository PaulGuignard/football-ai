"""Tests for the Deep Sets xG model, on hand-made shots and players."""

import numpy as np
import pandas as pd
import pytest
import torch
from test_xg import make_model_shots

from football_ai.deep_sets import (
    MAX_PLAYERS,
    PLAYER_FEATURES,
    DeepSetsNetwork,
    cross_validate_deep_sets,
    player_arrays,
    train_deep_sets,
)


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


def random_picture(n_shots=4, seed=0):
    """Random players for a few shots, each shot with a different number of them."""
    rng = np.random.default_rng(seed)
    positions = rng.normal(size=(n_shots, MAX_PLAYERS, len(PLAYER_FEATURES))).astype(np.float32)
    mask = np.zeros((n_shots, MAX_PLAYERS), dtype=np.float32)
    for shot in range(n_shots):
        mask[shot, : 3 + 4 * shot] = 1.0
    shot_inputs = rng.normal(size=(n_shots, 3)).astype(np.float32)
    return torch.from_numpy(positions), torch.from_numpy(mask), torch.from_numpy(shot_inputs)


def test_the_order_of_the_players_does_not_change_the_output():
    torch.manual_seed(0)
    network = DeepSetsNetwork(n_shot_inputs=3).eval()
    positions, mask, shot_inputs = random_picture()
    # Shuffle the 21 slots, the same way for the players and their mask
    order = torch.randperm(MAX_PLAYERS)

    with torch.no_grad():
        before = network(positions, mask, shot_inputs)
        after = network(positions[:, order], mask[:, order], shot_inputs)

    assert torch.allclose(before, after, atol=1e-5)


def test_empty_slots_are_ignored_whatever_they_contain():
    torch.manual_seed(0)
    network = DeepSetsNetwork(n_shot_inputs=3).eval()
    positions, mask, shot_inputs = random_picture()
    junk = positions.clone()
    junk[mask == 0] = 99.0

    with torch.no_grad():
        assert torch.allclose(network(positions, mask, shot_inputs),
                              network(junk, mask, shot_inputs), atol=1e-5)


def make_keeper_shots(n=1000, seed=0):
    """Fake shots seen with one keeper, who concedes more the further he is from goal."""
    rng = np.random.default_rng(seed)
    shots = make_model_shots(n, seed)
    positions = np.zeros((n, MAX_PLAYERS, len(PLAYER_FEATURES)), dtype=np.float32)
    mask = np.zeros((n, MAX_PLAYERS), dtype=np.float32)
    keeper_out = rng.uniform(0, 1, n)  # tens of yards off his line
    positions[:, 0, PLAYER_FEATURES.index("dx_goal")] = keeper_out
    positions[:, 0, PLAYER_FEATURES.index("keeper")] = 1.0
    mask[:, 0] = 1.0
    shots["is_goal"] = (rng.random(n) < 0.05 + 0.6 * keeper_out).astype(int)
    return shots, positions, mask, keeper_out


def test_the_network_learns_from_the_players():
    shots, positions, mask, keeper_out = make_keeper_shots()
    model = train_deep_sets(shots, positions, mask, max_epochs=60)
    xg = model.predict_xg(shots, positions, mask)

    assert xg.between(0, 1).all()
    assert xg[keeper_out > 0.7].mean() > xg[keeper_out < 0.3].mean() + 0.2


def test_the_same_seed_gives_the_same_model():
    shots, positions, mask, _ = make_keeper_shots(n=300)
    first = train_deep_sets(shots, positions, mask, max_epochs=5).predict_xg(shots, positions, mask)
    second = train_deep_sets(shots, positions, mask, max_epochs=5).predict_xg(shots, positions, mask)

    assert np.allclose(first, second)


def test_cross_validation_gives_one_log_loss_per_fold():
    shots, positions, mask, _ = make_keeper_shots(n=300)
    losses = cross_validate_deep_sets(shots, positions, mask, n_splits=3, max_epochs=3)

    assert len(losses) == 3
    assert np.isfinite(losses).all()