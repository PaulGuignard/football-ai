"""Tests for the demo helpers: the edits made on the pitch and the shot options."""

import math
import typing

import numpy as np
import pytest

from football_ai.api import PlayPattern
from football_ai.demo import (
    DEFAULT_PLAYERS,
    DEFAULT_SHOOTER,
    MAX_PLAYERS,
    PLAY_PATTERNS,
    allowed_spot,
    apply_edit,
    build_request,
    can_be_one_on_one,
    consistent_shot,
    describe,
    in_box,
    in_the_way,
    keep_in_heading_range,
    keep_out_of_box,
    rarity_note,
)
from football_ai.shot_inputs import in_shot_cone

NO_OPTIONS = {"is_header": False, "is_free_kick": False, "is_first_time": False,
              "is_one_on_one": False, "play_pattern": "Regular Play"}


def test_the_demo_offers_the_same_play_patterns_as_the_api():
    assert PLAY_PATTERNS == list(typing.get_args(PlayPattern))


def test_the_request_carries_the_shot_and_its_players():
    body = build_request(DEFAULT_SHOOTER, DEFAULT_PLAYERS, is_header=True)

    assert body["x"] == 108.0 and body["is_header"] is True
    assert len(body["players"]) == len(DEFAULT_PLAYERS)
    assert body["players"][0] == {"x": 119.0, "y": 40.0, "teammate": False, "keeper": True}


def test_moving_the_shooter():
    shooter, players = apply_edit(DEFAULT_SHOOTER, DEFAULT_PLAYERS,
                                  {"action": "move", "who": "shooter", "x": 100, "y": 30})

    assert shooter == {"x": 100.0, "y": 30.0}
    assert players == DEFAULT_PLAYERS


def test_moving_a_player_leaves_the_others_and_the_original_alone():
    _, players = apply_edit(DEFAULT_SHOOTER, DEFAULT_PLAYERS,
                            {"action": "move", "who": 1, "x": 115, "y": 38})

    assert players[1]["x"] == 115 and players[1]["y"] == 38
    assert players[0] == DEFAULT_PLAYERS[0]
    # The starting picture must stay as it was for "Start again"
    assert DEFAULT_PLAYERS[1]["x"] == 112.5


def test_bringing_a_defender_on():
    _, players = apply_edit(DEFAULT_SHOOTER, DEFAULT_PLAYERS,
                            {"action": "add", "kind": "defender", "x": 110, "y": 35})

    assert len(players) == len(DEFAULT_PLAYERS) + 1
    assert players[-1] == {"x": 110.0, "y": 35.0, "teammate": False, "keeper": False}


def test_a_new_keeper_takes_the_old_ones_place():
    _, players = apply_edit(DEFAULT_SHOOTER, DEFAULT_PLAYERS,
                            {"action": "add", "kind": "keeper", "x": 116, "y": 42})
    keepers = [p for p in players if p["keeper"] and not p["teammate"]]

    assert len(keepers) == 1 and keepers[0]["x"] == 116


def test_no_more_than_21_players():
    crowd = [{"x": 100.0, "y": 40.0, "teammate": True, "keeper": False}] * MAX_PLAYERS
    _, players = apply_edit(DEFAULT_SHOOTER, crowd,
                            {"action": "add", "kind": "defender", "x": 110, "y": 35})

    assert len(players) == MAX_PLAYERS


def test_taking_a_player_off():
    _, players = apply_edit(DEFAULT_SHOOTER, DEFAULT_PLAYERS, {"action": "remove", "who": 0})

    assert players == DEFAULT_PLAYERS[1:]


@pytest.mark.parametrize("edit", [
    {"action": "move", "who": "shooter", "x": 40, "y": 30},   # own half
    {"action": "move", "who": 7, "x": 110, "y": 30},          # nobody there
    {"action": "remove", "who": "shooter"},                   # he has to shoot
    {"action": "add", "kind": "referee", "x": 110, "y": 30},
    {"action": "dance"},
])
def test_edits_that_make_no_sense_change_nothing(edit):
    shooter, players = apply_edit(DEFAULT_SHOOTER, DEFAULT_PLAYERS, edit)

    assert shooter == DEFAULT_SHOOTER and players == DEFAULT_PLAYERS


def test_a_direct_free_kick_rules_out_what_it_never_comes_with():
    shot = consistent_shot({key: True for key in NO_OPTIONS} | {"play_pattern": "From Corner"})

    assert shot["is_free_kick"] is True
    assert not any(shot[k] for k in ("is_header", "is_first_time", "is_one_on_one"))
    assert shot["play_pattern"] == "From Free Kick"


def test_a_header_is_not_first_time_but_can_follow_a_corner():
    shot = consistent_shot(NO_OPTIONS | {"is_header": True, "is_first_time": True,
                                         "is_one_on_one": True, "play_pattern": "From Corner"})

    assert shot["is_first_time"] is False
    assert shot["is_one_on_one"] is True and shot["play_pattern"] == "From Corner"


def test_options_that_go_together_are_kept():
    shot = NO_OPTIONS | {"is_first_time": True, "is_one_on_one": True}

    assert consistent_shot(shot) == shot


def test_the_penalty_spot_is_in_the_box_and_its_lines_are_not():
    assert in_box(108, 40)
    assert not in_box(102, 40) and not in_box(110, 18) and not in_box(95, 40)


@pytest.mark.parametrize("spot, out", [
    ({"x": 103.0, "y": 40.0}, {"x": 102.0, "y": 40.0}),   # nearest way out: the front
    ({"x": 115.0, "y": 20.0}, {"x": 115.0, "y": 18.0}),   # one side
    ({"x": 115.0, "y": 61.0}, {"x": 115.0, "y": 62.0}),   # the other side
    ({"x": 95.0, "y": 40.0}, {"x": 95.0, "y": 40.0}),     # already outside
])
def test_a_spot_in_the_box_goes_to_its_nearest_edge(spot, out):
    assert keep_out_of_box(spot) == out


def test_a_free_kick_shooter_cant_be_dragged_into_the_box():
    edit = {"action": "move", "who": "shooter", "x": 108, "y": 40}
    shooter, _ = apply_edit(DEFAULT_SHOOTER, DEFAULT_PLAYERS, edit, free_kick=True)

    assert not in_box(shooter["x"], shooter["y"])
    # Without a free kick the same move is fine
    assert apply_edit(DEFAULT_SHOOTER, DEFAULT_PLAYERS, edit)[0] == {"x": 108.0, "y": 40.0}


def test_a_header_from_too_far_is_pulled_back_towards_goal():
    spot = keep_in_heading_range({"x": 60.0, "y": 40.0})   # 60 yards out

    assert spot == pytest.approx({"x": 90.0, "y": 40.0})    # 30 yards out, same line
    assert keep_in_heading_range({"x": 100.0, "y": 30.0}) == {"x": 100.0, "y": 30.0}


def test_a_header_cant_be_dragged_far_from_goal():
    edit = {"action": "move", "who": "shooter", "x": 70, "y": 20}
    shooter, _ = apply_edit(DEFAULT_SHOOTER, DEFAULT_PLAYERS, edit, header=True)

    assert math.dist((shooter["x"], shooter["y"]), (120, 40)) == pytest.approx(30)


def test_allowed_spot_applies_only_the_ticked_options():
    in_the_box = {"x": 110.0, "y": 40.0}

    assert allowed_spot(in_the_box) == in_the_box
    assert allowed_spot(in_the_box, header=True) == in_the_box
    assert not in_box(**allowed_spot(in_the_box, free_kick=True))


def test_a_warning_only_for_shots_from_very_far():
    assert rarity_note(25) == ""
    assert "very few shots" in rarity_note(45)


def test_the_last_defender_cant_be_taken_off():
    # Index 1 is the only defender of the starting picture
    _, players = apply_edit(DEFAULT_SHOOTER, DEFAULT_PLAYERS, {"action": "remove", "who": 1})
    assert players == DEFAULT_PLAYERS

    # With a second one on, either can go
    two = DEFAULT_PLAYERS + [{"x": 100.0, "y": 30.0, "teammate": False, "keeper": False}]
    _, players = apply_edit(DEFAULT_SHOOTER, two, {"action": "remove", "who": 1})
    assert len(players) == len(two) - 1


def test_the_demo_and_the_api_agree_on_who_is_in_the_way():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        shooter = {"x": rng.uniform(70, 119), "y": rng.uniform(0, 80)}
        player = {"x": rng.uniform(70, 120), "y": rng.uniform(0, 80)}
        expected = in_shot_cone(player["x"], player["y"], shooter["x"], shooter["y"])
        assert in_the_way(player, shooter) == expected


def test_no_one_on_one_with_a_defender_in_the_way():
    shooter = {"x": 108.0, "y": 40.0}
    in_front = {"x": 113.0, "y": 40.0, "teammate": False, "keeper": False}

    assert not can_be_one_on_one(shooter, [in_front])
    assert can_be_one_on_one(shooter, [in_front | {"y": 60.0}])        # off to the side
    assert can_be_one_on_one(shooter, [in_front | {"teammate": True}])  # one of his own
    assert can_be_one_on_one(shooter, [in_front | {"keeper": True}])    # the keeper
    shot = NO_OPTIONS | {"is_one_on_one": True}
    assert consistent_shot(shot, one_on_one_possible=False)["is_one_on_one"] is False


@pytest.mark.parametrize("xg, words", [
    (0.384, "About 38 shots in 100 like this one go in."),
    (0.012, "About 1 shot in 100 like this one goes in."),
    (0.002, "Fewer than 1 shot in 100 like this one goes in."),
])
def test_xg_in_words(xg, words):
    assert describe(xg) == words