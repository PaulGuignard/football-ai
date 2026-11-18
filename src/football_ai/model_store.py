"""Saving the xG model to a file, and loading it back for the API.

The API never trains anything and has no database: it loads this file when it
starts. skops is used rather than pickle because a pickle file can run any
code when it is opened, while skops only rebuilds the types it is told to
trust.
"""

import json
from pathlib import Path

import sklearn
import skops.io as sio
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xg_model.skops"
# Types skops won't reload unless told to: the boosting's trees, and the
# numpy dtype the imputer keeps to remember what it was trained on
TRUSTED_TYPES = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "numpy.dtype",
]


def save_model(model: Pipeline, path: Path = MODEL_PATH, **details) -> None:
    """Write the model, plus a small JSON file next to it describing it.

    The JSON records the scikit-learn version, since a model should be loaded
    with the version that trained it, and whatever details are passed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(model, path)
    info = {"scikit_learn": sklearn.__version__, **details}
    path.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Read a model written by save_model."""
    return sio.load(path, trusted=TRUSTED_TYPES)