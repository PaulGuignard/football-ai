"""Tests for saving and loading the model file used by the API."""

import json

from test_xg import make_v2_shots

from football_ai.model_store import load_model, save_model
from football_ai.xg import build_boosting_model, build_logistic_model, predict_xg, train


def test_a_saved_model_gives_back_the_same_xg(tmp_path):
    shots = make_v2_shots()
    path = tmp_path / "models" / "xg_model.skops"
    for build_model in (build_boosting_model, build_logistic_model):
        model = train(build_model("v2"), shots)
        save_model(model, path)

        assert predict_xg(load_model(path), shots).equals(predict_xg(model, shots))


def test_the_details_are_written_next_to_the_model(tmp_path):
    path = tmp_path / "xg_model.skops"
    save_model(train(build_boosting_model("v2"), make_v2_shots()), path, shots=400)
    details = json.loads(path.with_suffix(".json").read_text())

    assert details["shots"] == 400
    assert "scikit_learn" in details