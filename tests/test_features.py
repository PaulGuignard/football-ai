"""Tests for the shot_features table, on a few hand-made shots.

Each test builds a tiny database in memory, so nothing depends on the real
data and the whole file runs in under a second.
"""

import math

import duckdb
import pytest

from football_ai.features import SCHEMA_PATH, build_shot_features


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