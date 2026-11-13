"""Train both xG models and record each one as a run in MLflow.

Same split as the evaluation script: the scores are the same, but this time
they are kept, with the settings and the trained model.

Usage:
    uv run python scripts/09_track_xg.py
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
Then open http://127.0.0.1:5000 in a browser.
"""

from pathlib import Path

import duckdb
import pandas as pd

from football_ai.tracking import log_xg_run, use_experiment
from football_ai.xg import (
    build_boosting_model,
    build_logistic_model,
    load_shots,
    split_by_match,
    train,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "football.duckdb"


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        shots = load_shots(con)
    train_set, test_set = split_by_match(shots)

    use_experiment()
    scores = {}
    for run_name, build_model in [
        ("logistic regression", build_logistic_model),
        ("gradient boosting", build_boosting_model),
    ]:
        model = train(build_model(), train_set)
        scores[run_name] = log_xg_run(run_name, model, train_set, test_set)
        print(f"Logged run: {run_name}")

    print()
    print(pd.DataFrame(scores).T.round(4).to_string())


if __name__ == "__main__":
    main()