"""Tests for the Python version of the model inputs, used by the API.

The last test is the important one: on random shots and players, the Python
inputs must match the SQL ones that the model was trained on.
"""

import math

import duckdb
import numpy as np
import pandas as pd
import pytest
from test_xg import make_v2_shots

from football_ai.features import (
    SCHEMA_PATH,
    build_shot_features,
    build_shot_freeze_features,
)
from football_ai.shot_inputs import distance_and_angle, freeze_inputs, shot_row
from football_ai.xg import (
    FEATURE_SETS,
    FREEZE_FEATURES,
    build_boosting_model,
    predict_xg,
    train,
)


def player(x, y, teammate=False, keeper=False):
    """One player as the API receives it."""
    return {"x": x, "y": y, "teammate": teammate, "keeper": keeper}


def test_penalty_spot_geometry():
    distance, angle = distance_and_angle(108.0, 40.0)

    assert distance == pytest.approx(12.0)
    assert angle == pytest.approx(math.degrees(2 * math.atan(4 / 12)))


def test_only_opponents_between_ball_and_goal_count_as_blocking():
    inputs = freeze_inputs(108.0, 40.0, [
        player(114, 40),  # right in the way
        player(100, 40),  # behind the shooter
        player(114, 60),  # well wide of the posts
        player(115, 41, teammate=True),
    ])

    assert inputs["defenders_in_cone"] == 1


def test_keeper_out_of_position_and_nearest_defender():
    inputs = freeze_inputs(108.0, 40.0, [player(116, 30, keeper=True), player(111, 44)])

    assert inputs["nearest_defender"] == pytest.approx(5.0)
    assert inputs["keeper_to_goal"] == pytest.approx(math.hypot(4, 10))
    assert inputs["keeper_in_cone"] == 0


def test_nobody_or_two_keepers_leave_the_inputs_missing():
    nobody = freeze_inputs(108.0, 40.0, [])
    two_keepers = freeze_inputs(108.0, 40.0, [player(119, 40, keeper=True),
                                              player(118, 42, keeper=True)])

    assert nobody["defenders_in_cone"] == 0
    assert math.isnan(nobody["nearest_defender"])
    assert math.isnan(two_keepers["keeper_to_goal"])


def test_a_shot_row_feeds_a_trained_v2_model():
    model = train(build_boosting_model("v2"), make_v2_shots())
    row = shot_row(108.0, 40.0, [player(119, 40, keeper=True)], is_header=True)

    assert list(row.columns) == FEATURE_SETS["v2"]
    assert 0 < predict_xg(model, row).iloc[0] < 1


def test_python_inputs_match_the_sql_ones_on_random_shots():
    rng = np.random.default_rng(0)
    con = duckdb.connect()
    con.execute(SCHEMA_PATH.read_text())
    con.execute("INSERT INTO matches (match_id, competition) VALUES (1, 'Test League')")

    expected = {}
    for i in range(300):
        shot_id = f"s{i}"
        sx, sy = rng.uniform(80, 119), rng.uniform(10, 70)
        con.execute(
            "INSERT INTO events (match_id, event_id, type, x, y, shot_type, shot_outcome) "
            "VALUES (1, ?, 'Shot', ?, ?, 'Open Play', 'Saved')",
            [shot_id, sx, sy],
        )
        players = []
        for _ in range(rng.integers(0, 12)):
            # Some players sit exactly on the line from the ball to a post
            if rng.random() < 0.2:
                t = rng.uniform(0, 1)
                x, y = sx + t * (120 - sx), sy + t * (rng.choice([36.0, 44.0]) - sy)
            else:
                x, y = rng.uniform(80, 120), rng.uniform(0, 80)
            p = player(x, y, teammate=rng.random() < 0.3, keeper=rng.random() < 0.15)
            players.append(p)
            con.execute(
                "INSERT INTO shot_freeze_frames (shot_id, match_id, position, teammate, x, y) "
                "VALUES (?, 1, ?, ?, ?, ?)",
                [shot_id, "Goalkeeper" if p["keeper"] else "Center Back", p["teammate"], x, y],
            )
        expected[shot_id] = shot_row(sx, sy, players).iloc[0]

    build_shot_features(con)
    build_shot_freeze_features(con)
    sql = con.execute("""
        SELECT * FROM shot_features JOIN shot_freeze_features USING (shot_id)
    """).df().set_index("shot_id")

    for shot_id, python in expected.items():
        for column in ["distance", "angle", *FREEZE_FEATURES]:
            sql_value, python_value = sql.loc[shot_id, column], python[column]
            if pd.isna(python_value):
                # Missing in Python, so it must be missing in SQL too
                assert pd.isna(sql_value), (shot_id, column)
            else:
                assert float(sql_value) == pytest.approx(python_value, abs=1e-9), (shot_id, column)