"""Experiment tracking with MLflow: every training run leaves a trace.

Each run records what was chosen (model, settings, inputs), how it scored,
and for test runs the trained model itself, so any result can be found and
compared later in the MLflow UI. Two kinds of runs: test runs (a trained
model scored once on the test matches) and cross-validation runs (a setting
scored on folds of the training matches, to choose between settings). The
Deep Sets network has its own pair of functions to save and reload it.
"""

import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from football_ai.deep_sets import DeepSetsNetwork, DeepSetsXG
from football_ai.model_store import TRUSTED_TYPES
from football_ai.xg import evaluate, model_inputs, predict_xg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Results, not code: both are in .gitignore and can be rebuilt by re-running
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
ARTIFACTS_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT = "xg"


def use_experiment(
    name: str = EXPERIMENT,
    tracking_uri: str = TRACKING_URI,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> None:
    """Point MLflow at the project's tracking database and pick the experiment.

    The experiment is created the first time. Its files (saved models) go to
    artifacts_dir, wherever the script is launched from.
    """
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifacts_dir.as_uri())
    mlflow.set_experiment(name)


def log_xg_run(
    run_name: str,
    model: Pipeline,
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    feature_set: str = "v1",
) -> dict[str, float]:
    """Score a trained xG model on the test matches and record it as one MLflow run.

    feature_set names the inputs the model saw ("v1": the eight shot inputs,
    "v2": the same plus the freeze frame ones), so that runs with and without
    player positions can be told apart.
    Returns the scores, so the caller can print them.
    """
    scores = evaluate(test_set["is_goal"], predict_xg(model, test_set))

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("stage", "test")
        estimator = model.named_steps["model"]
        mlflow.log_params({
            "model": type(estimator).__name__,
            "feature_set": feature_set,
            "n_features": len(model_inputs(model)),
            "train_shots": len(train_set),
            "test_shots": len(test_set),
        })
        # Every setting of the model, defaults included: nothing left to guess
        mlflow.log_params(estimator.get_params())
        mlflow.log_metrics(scores)
        if isinstance(estimator, HistGradientBoostingClassifier):
            # Not chosen by us: how many trees early stopping kept
            mlflow.log_metric("n_trees", estimator.n_iter_)
        # MLflow saves models with skops, which refuses to reload some types
        # unless told to: the file could be tampered with. We made it, so we
        # trust the few types listed above, not everything
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=TRUSTED_TYPES)

    return scores


def log_cv_run(
    run_name: str,
    settings: dict,
    fold_losses: np.ndarray,
    references: dict[str, np.ndarray] | None = None,
    feature_set: str = "v1",
) -> None:
    """Record one setting's cross-validation scores as an MLflow run.

    Keeps the mean and spread of the fold log losses, and each fold's own
    score. references maps a name to another model's fold scores (e.g.
    {"logistic": ...}): for each one, counts on how many folds this setting
    did better. The folds are the same for every model, so this fold-by-fold
    count is a fairer test than comparing means.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("stage", "cross-validation")
        mlflow.log_params({**settings, "feature_set": feature_set, "n_folds": len(fold_losses)})
        mlflow.log_metrics({
            "cv_log_loss": float(fold_losses.mean()),
            "cv_log_loss_std": float(fold_losses.std()),
        })
        for fold, loss in enumerate(fold_losses):
            # One metric, one value per fold: MLflow draws it as a small curve
            mlflow.log_metric("fold_log_loss", float(loss), step=fold)
        for name, reference in (references or {}).items():
            mlflow.log_metric(f"folds_beating_{name}", int((fold_losses < reference).sum()))


def log_deep_sets_run(
    run_name: str,
    model: DeepSetsXG,
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    test_positions: np.ndarray,
    test_mask: np.ndarray,
    settings: dict | None = None,
) -> dict[str, float]:
    """Score a trained Deep Sets network on the test matches and record it as one run.

    The weights are saved as a plain PyTorch file and the preparation of the
    shot inputs as a scikit-learn model, so load_deep_sets can rebuild it.
    Returns the scores.
    """
    xg = model.predict_xg(test_set, test_positions, test_mask)
    scores = evaluate(test_set["is_goal"], xg)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("stage", "test")
        mlflow.log_params({
            **(settings or {}),
            "model": "DeepSets",
            "feature_set": "raw positions",
            "hidden": model.network.encode_player[0].out_features,
            "train_shots": len(train_set),
            "test_shots": len(test_set),
        })
        mlflow.log_metrics({**scores, "best_epoch": model.best_epoch})
        # The weights alone, not the whole Python object: a weights file can be
        # read back without running any code stored in it
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder) / "network.pt"
            torch.save(model.network.state_dict(), weights)
            mlflow.log_artifact(str(weights), artifact_path="deep_sets")
        mlflow.sklearn.log_model(
            model.preparation, name="preparation", skops_trusted_types=TRUSTED_TYPES
        )

    return scores


def load_deep_sets(run_id: str) -> DeepSetsXG:
    """Rebuild a Deep Sets network saved by log_deep_sets_run."""
    params = mlflow.get_run(run_id).data.params
    preparation = mlflow.sklearn.load_model(f"runs:/{run_id}/preparation")
    network = DeepSetsNetwork(
        n_shot_inputs=len(preparation.get_feature_names_out()), hidden=int(params["hidden"])
    )
    weights = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="deep_sets/network.pt")
    network.load_state_dict(torch.load(weights, weights_only=True))
    best_epoch = int(mlflow.get_run(run_id).data.metrics["best_epoch"])
    return DeepSetsXG(preparation, network, best_epoch)