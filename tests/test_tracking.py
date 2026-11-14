"""Tests for MLflow tracking, against a throwaway tracking database."""

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from test_xg import make_model_shots

from football_ai.tracking import log_cv_run, log_xg_run, use_experiment
from football_ai.xg import build_boosting_model, predict_xg, split_by_match, train


@pytest.fixture
def experiment(tmp_path):
    """A fresh experiment in a temporary folder, never the project's own database."""
    use_experiment(
        name="test",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifacts_dir=tmp_path / "mlruns",
    )


def test_a_run_keeps_settings_scores_and_model(experiment):
    train_set, test_set = split_by_match(make_model_shots())
    model = train(build_boosting_model(), train_set)
    scores = log_xg_run("boosting", model, train_set, test_set)

    run = mlflow.search_runs(experiment_names=["test"]).iloc[0]
    assert run["tags.mlflow.runName"] == "boosting"
    assert run["tags.stage"] == "test"
    assert run["params.model"] == "HistGradientBoostingClassifier"
    assert run["params.learning_rate"] == "0.1"
    assert run["metrics.log_loss"] == pytest.approx(scores["log_loss"])
    assert "metrics.n_trees" in run

    # The saved model gives back exactly the same xG
    reloaded = mlflow.sklearn.load_model(f"runs:/{run['run_id']}/model")
    assert predict_xg(reloaded, test_set).equals(predict_xg(model, test_set))


def test_a_cv_run_keeps_the_setting_and_every_fold(experiment):
    losses = np.array([0.25, 0.27, 0.26])
    default = np.array([0.26, 0.26, 0.27])
    log_cv_run("cv small trees", {"max_leaf_nodes": 7}, losses, {"default": default})

    run = mlflow.search_runs(experiment_names=["test"]).iloc[0]
    assert run["tags.stage"] == "cross-validation"
    assert run["params.max_leaf_nodes"] == "7"
    assert run["metrics.cv_log_loss"] == pytest.approx(0.26)
    # Better on the first and last folds, worse on the middle one
    assert run["metrics.folds_beating_default"] == 2

    history = mlflow.MlflowClient().get_metric_history(run["run_id"], "fold_log_loss")
    assert [m.value for m in sorted(history, key=lambda m: m.step)] == [0.25, 0.27, 0.26]