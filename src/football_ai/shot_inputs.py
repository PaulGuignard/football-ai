"""The model inputs of one shot, computed in Python for the API.

During training these inputs come from SQL (sql/features/). The API has no
database, only the positions it is sent, so the same formulas are written
again here. Any difference would feed the model numbers it never learnt from
without anything crashing, so tests check that both versions agree.
"""

import math

import pandas as pd

from football_ai.xg import FEATURE_SETS

GOAL_X, GOAL_Y = 120.0, 40.0
POST_LOW, POST_HIGH = 36.0, 44.0


def distance_and_angle(x: float, y: float) -> tuple[float, float]:
    """Distance to the centre of the goal, and how many degrees of goal the shooter sees."""
    distance = math.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)
    angle = math.degrees(abs(
        math.atan2(POST_HIGH - y, GOAL_X - x) - math.atan2(POST_LOW - y, GOAL_X - x)
    ))
    return distance, angle


def side(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Which side of the line A -> B the point P is on, as in the SQL macro."""
    # Same operations in the same order as SQL, so both round the same way
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def in_shot_cone(px: float, py: float, shot_x: float, shot_y: float) -> bool:
    """Is P inside the triangle between the ball and the two posts?"""
    edges = [
        side(px, py, shot_x, shot_y, GOAL_X, POST_LOW),
        side(px, py, GOAL_X, POST_LOW, GOAL_X, POST_HIGH),
        side(px, py, GOAL_X, POST_HIGH, shot_x, shot_y),
    ]
    # On the edge counts as inside, like >= 0 and <= 0 in SQL
    return all(e >= 0 for e in edges) or all(e <= 0 for e in edges)


def freeze_inputs(shot_x: float, shot_y: float, players: list[dict]) -> dict[str, float]:
    """The four freeze frame inputs of a shot, from the players around it.

    Each player is a dict with x, y, teammate and keeper (both booleans).
    Teammates are ignored. Missing values are NaN, as in the training data:
    no outfield opponent means no nearest defender, and anything but exactly
    one opposing keeper means no keeper inputs.
    """
    opponents = [p for p in players if not p["teammate"]]
    keepers = [p for p in opponents if p["keeper"]]
    outfield = [p for p in opponents if not p["keeper"]]

    distances = [math.sqrt((p["x"] - shot_x) ** 2 + (p["y"] - shot_y) ** 2) for p in outfield]
    inputs = {
        "defenders_in_cone": float(sum(in_shot_cone(p["x"], p["y"], shot_x, shot_y)
                                       for p in outfield)),
        "nearest_defender": min(distances) if distances else math.nan,
        "keeper_to_goal": math.nan,
        "keeper_in_cone": math.nan,
    }
    if len(keepers) == 1:
        keeper = keepers[0]
        inputs["keeper_to_goal"] = math.sqrt((GOAL_X - keeper["x"]) ** 2 + (GOAL_Y - keeper["y"]) ** 2)
        inputs["keeper_in_cone"] = float(in_shot_cone(keeper["x"], keeper["y"], shot_x, shot_y))
    return inputs


def shot_row(
    x: float,
    y: float,
    players: list[dict] = (),
    is_header: bool = False,
    is_free_kick: bool = False,
    is_first_time: bool = False,
    is_one_on_one: bool = False,
    is_under_pressure: bool = False,
    play_pattern: str = "Regular Play",
) -> pd.DataFrame:
    """One shot as a one-row table with every v2 input, ready for the model."""
    distance, angle = distance_and_angle(x, y)
    row = {
        "distance": distance,
        "angle": angle,
        "is_header": is_header,
        "is_free_kick": is_free_kick,
        "is_first_time": is_first_time,
        "is_one_on_one": is_one_on_one,
        "is_under_pressure": is_under_pressure,
        "play_pattern": play_pattern,
        **freeze_inputs(x, y, list(players)),
    }
    return pd.DataFrame([row])[FEATURE_SETS["v2"]]