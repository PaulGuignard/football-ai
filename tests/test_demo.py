"""Tests for the demo helpers: the edits made on the pitch and the shot options."""

import typing

import pytest

from football_ai.api import PlayPattern
from football_ai.demo import (
    DEFAULT_PLAYERS,
    DEFAULT_SHOOTER,
    MAX_PLAYERS,
    PLAY_PATTERNS,
    apply_edit,
    build_request,
    consistent_shot,
    describe,
)

NO_OPTIONS = {"is_header": False, "is_free_kick": False, "is_first_time": False,
              "is_one_on_one": False, "is_under_pressure": False,
              "play_pattern": "Regular Play"}


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
    assert DEFAULT_PLAYERS[1]["x"] == 113.0


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
    assert not any(shot[k] for k in ("is_header", "is_first_time", "is_one_on_one",
                                     "is_under_pressure"))
    assert shot["play_pattern"] == "From Free Kick"


def test_a_header_is_not_first_time_but_can_follow_a_corner():
    shot = consistent_shot(NO_OPTIONS | {"is_header": True, "is_first_time": True,
                                         "is_one_on_one": True, "play_pattern": "From Corner"})

    assert shot["is_first_time"] is False
    assert shot["is_one_on_one"] is True and shot["play_pattern"] == "From Corner"


def test_options_that_go_together_are_kept():
    shot = NO_OPTIONS | {"is_first_time": True, "is_under_pressure": True}

    assert consistent_shot(shot) == shot


@pytest.mark.parametrize("xg, words", [
    (0.384, "About 38 shots in 100 like this one go in."),
    (0.012, "About 1 shot in 100 like this one goes in."),
    (0.002, "Fewer than 1 shot in 100 like this one goes in."),
])
def test_xg_in_words(xg, words):
    assert describe(xg) == words