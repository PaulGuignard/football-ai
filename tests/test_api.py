"""Tests for the xG API, against the real model file in models/."""

import pytest
from fastapi.testclient import TestClient

from football_ai.api import app

KEEPER = {"x": 119.0, "y": 40.0, "keeper": True}


@pytest.fixture(scope="module")
def client():
    # The with block runs the startup, which loads the model
    with TestClient(app) as client:
        yield client


def test_health_says_which_model_is_served(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model"]["model_name"] == "gradient boosting v2"


def test_a_shot_gets_an_xg_and_its_inputs(client):
    response = client.post("/xg", json={"x": 108, "y": 40, "players": [KEEPER]})
    body = response.json()

    assert response.status_code == 200
    assert 0 < body["xg"] < 1
    assert body["inputs"]["distance"] == pytest.approx(12.0)
    assert body["inputs"]["keeper_in_cone"] == 1


def test_a_defender_in_the_way_lowers_the_xg(client):
    open_goal = client.post("/xg", json={"x": 108, "y": 40, "players": [KEEPER]}).json()
    blocked = client.post("/xg", json={
        "x": 108, "y": 40, "players": [KEEPER, {"x": 112, "y": 40}],
    }).json()

    assert blocked["xg"] < open_goal["xg"]


def test_unknown_inputs_come_back_as_null(client):
    # Nobody in the picture: no nearest defender and no keeper
    body = client.post("/xg", json={"x": 108, "y": 40}).json()

    assert body["inputs"]["nearest_defender"] is None
    assert body["inputs"]["keeper_to_goal"] is None


@pytest.mark.parametrize("bad_shot", [
    {"x": 500, "y": 40},  # off the pitch
    {"x": 108},  # no y
    {"x": 108, "y": 40, "play_pattern": "From Penalty"},  # not a StatsBomb pattern
    {"x": 108, "y": 40, "players": [{"x": 110, "y": 40}] * 22},  # too many players
])
def test_bad_shots_are_refused_with_a_clear_error(client, bad_shot):
    response = client.post("/xg", json=bad_shot)

    # 422: the request is understood but its content is wrong
    assert response.status_code == 422