"""Experiment tracking with MLflow: every training run leaves a trace.

Each run records what was chosen (model, settings, inputs), how it scored on
the test matches, and the trained model itself, so any result can be found
and compared later in the MLflow UI.
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from football_ai.xg import FEATURES, evaluate, predict_xg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Results, not code: both are in .gitignore and can be rebuilt by re-running
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
ARTIFACTS_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT = "xg"
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


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

    feature_set names the inputs the model saw ("v1": the eight shot inputs),
    so that runs with and without player positions can be told apart later.
    Returns the scores, so the caller can print them.
    """
    scores = evaluate(test_set["is_goal"], predict_xg(model, test_set))

    with mlflow.start_run(run_name=run_name):
        estimator = model.named_steps["model"]
        mlflow.log_params({
            "model": type(estimator).__name__,
            "feature_set": feature_set,
            "n_features": len(FEATURES),
            "train_shots": len(train_set),
            "test_shots": len(test_set),
        })
        # Every setting of the model, defaults included: nothing left to guess
        mlflow.log_params(estimator.get_params())
        mlflow.log_metrics(scores)
        if isinstance(estimator, HistGradientBoostingClassifier):
            # Not chosen by us: how many trees early stopping kept
            mlflow.log_metric("n_trees", estimator.n_iter_)
        # MLflow saves models with skops, which refuses to reload the boosting
        # trees unless told to: the file could be tampered with. We made it, so
        # we trust that one type, not everything
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=TRUSTED_TYPES)

    return scores