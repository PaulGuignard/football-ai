"""Tests for the shot_features and shot_freeze_features tables, on hand-made shots.

Each test builds a tiny database in memory, so nothing depends on the real
data and the whole file runs in under a second.
"""

import math

import duckdb
import pandas as pd
import pytest

from football_ai.features import (
    SCHEMA_PATH,
    build_shot_features,
    build_shot_freeze_features,
)
from football_ai.xg import FREEZE_FEATURES, load_shots


@pytest.fixture
def con():
    """An empty in-memory database with the project schema and one match."""
    con = duckdb.connect()
    con.execute(SCHEMA_PATH.read_text())
    con.execute("INSERT INTO matches (match_id, competition) VALUES (1, 'Test League')")
    yield con
    con.close()


def add_shot(con, event_id, x=108.0, y=40.0, outcome="Saved", shot_type="Open Play", **extra):
    """Insert one shot; by default an open-play shot from the penalty spot."""
    values = {
        "match_id": 1, "event_id": event_id, "type": "Shot",
        "x": x, "y": y, "shot_outcome": outcome, "shot_type": shot_type, **extra,
    }
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    con.execute(f"INSERT INTO events ({columns}) VALUES ({placeholders})", list(values.values()))


def test_penalty_spot_distance_and_angle(con):
    add_shot(con, "s1")
    build_shot_features(con)
    distance, angle = con.execute("SELECT distance, angle FROM shot_features").fetchone()

    # The spot is 12 yards out and the posts are 4 yards each side of centre
    assert distance == pytest.approx(12.0)
    assert angle == pytest.approx(math.degrees(2 * math.atan(4 / 12)))  # about 36.9


def test_penalties_corners_and_other_events_are_left_out(con):
    add_shot(con, "open_play")
    add_shot(con, "penalty", shot_type="Penalty")
    add_shot(con, "corner", shot_type="Corner")
    con.execute("INSERT INTO events (match_id, event_id, type) VALUES (1, 'a_pass', 'Pass')")

    assert build_shot_features(con) == 1


def test_goal_is_1_and_anything_else_is_0(con):
    add_shot(con, "goal", outcome="Goal")
    add_shot(con, "saved", outcome="Saved")
    add_shot(con, "post", outcome="Post")
    build_shot_features(con)
    rows = dict(con.execute("SELECT shot_id, is_goal FROM shot_features").fetchall())

    assert rows == {"goal": 1, "saved": 0, "post": 0}


def test_header_and_free_kick_flags(con):
    add_shot(con, "header", shot_body_part="Head")
    add_shot(con, "free_kick", shot_type="Free Kick", shot_body_part="Right Foot")
    build_shot_features(con)
    rows = {
        shot_id: (is_header, is_free_kick)
        for shot_id, is_header, is_free_kick in con.execute(
            "SELECT shot_id, is_header, is_free_kick FROM shot_features"
        ).fetchall()
    }

    assert rows == {"header": (True, False), "free_kick": (False, True)}


def add_player(con, shot_id, x, y, teammate=False, position="Center Back"):
    """Put one player in a shot's freeze frame."""
    con.execute(
        "INSERT INTO shot_freeze_frames (shot_id, match_id, position, teammate, x, y) "
        "VALUES (?, 1, ?, ?, ?, ?)",
        [shot_id, position, teammate, x, y],
    )


def freeze_features(con):
    """Build both tables and return the freeze inputs of each shot, by shot_id."""
    build_shot_features(con)
    build_shot_freeze_features(con)
    rows = con.execute("SELECT * FROM shot_freeze_features").df()
    return rows.set_index("shot_id")


def test_only_opponents_between_ball_and_goal_count_as_blocking(con):
    add_shot(con, "s1")  # from the penalty spot, (108, 40)
    add_player(con, "s1", 114, 40)  # right in the way
    add_player(con, "s1", 100, 40)  # behind the shooter
    add_player(con, "s1", 114, 60)  # well wide of the posts
    add_player(con, "s1", 115, 41, teammate=True)  # in the way, but a teammate

    assert freeze_features(con).loc["s1", "defenders_in_cone"] == 1


def test_nearest_defender_ignores_the_keeper(con):
    add_shot(con, "s1")
    add_player(con, "s1", 119, 40, position="Goalkeeper")
    add_player(con, "s1", 111, 44)  # 5 yards away, a 3-4-5 triangle
    add_player(con, "s1", 100, 40)

    assert freeze_features(con).loc["s1", "nearest_defender"] == pytest.approx(5.0)


def test_keeper_on_his_line_in_the_middle(con):
    add_shot(con, "s1")
    add_player(con, "s1", 120, 40, position="Goalkeeper")
    row = freeze_features(con).loc["s1"]

    assert row["keeper_to_goal"] == pytest.approx(0.0)
    assert row["keeper_in_cone"] == 1


def test_keeper_caught_out_of_position(con):
    add_shot(con, "s1")
    add_player(con, "s1", 116, 30, position="Goalkeeper")
    row = freeze_features(con).loc["s1"]

    assert row["keeper_to_goal"] == pytest.approx(math.hypot(4, 10))
    assert row["keeper_in_cone"] == 0


def test_shot_with_nobody_in_the_picture_still_gets_a_row(con):
    add_shot(con, "s1")
    row = freeze_features(con).loc["s1"]

    assert row["defenders_in_cone"] == 0
    # Unknown, not zero: the model will have to fill these in
    assert pd.isna(row["nearest_defender"])
    assert pd.isna(row["keeper_to_goal"])


@pytest.mark.parametrize("freeze_frames", [False, True])
def test_shots_always_load_in_the_same_order(con, freeze_frames):
    for event_id in ["s3", "s1", "s2"]:
        add_shot(con, event_id)
    freeze_features(con)

    assert load_shots(con, freeze_frames)["shot_id"].tolist() == ["s1", "s2", "s3"]


def test_loaded_freeze_inputs_are_plain_numbers(con):
    # One shot with a keeper, one with nobody in the picture: gaps to handle
    add_shot(con, "s1")
    add_player(con, "s1", 120, 40, position="Goalkeeper")
    add_shot(con, "s2")
    freeze_features(con)
    shots = load_shots(con, freeze_frames=True).set_index("shot_id")

    assert (shots[FREEZE_FEATURES].dtypes == "float64").all()
    assert pd.isna(shots.loc["s2", "keeper_in_cone"])