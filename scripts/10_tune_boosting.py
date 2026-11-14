"""Choose the gradient boosting settings by cross-validation, then test the winner once.

Every combination in GRID is scored on 5 folds of the training matches and
logged in MLflow, next to two references scored on the same folds: the
boosting's default settings and the logistic regression. The combination
with the lowest mean log loss is retrained on all the training matches and
scored on the test matches, the only time the test set is used.

Usage:
    uv run python scripts/10_tune_boosting.py
"""

from itertools import product
from pathlib import Path

import duckdb
import pandas as pd

from football_ai.tracking import log_cv_run, log_xg_run, use_experiment
from football_ai.xg import (
    build_boosting_model,
    build_logistic_model,
    cross_validate_by_match,
    load_shots,
    split_by_match,
    train,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "football.duckdb"

# 2 x 5 x 2 = 20 combinations, 100 trainings in all. Trees go down to 2
# leaves: if the smallest size won, we'd want to know whether even smaller does
GRID = {
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [2, 4, 7, 15, 31],
    "min_samples_leaf": [20, 100],
}
# scikit-learn's own choices: does tuning beat leaving the defaults alone?
DEFAULT = {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 20}
# A slower learning rate needs more trees; early stopping still decides when to stop
MAX_TREES = 1000


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        shots = load_shots(con)
    train_set, test_set = split_by_match(shots)

    # The other reference: is the boosting worth it over the simple model?
    logistic_losses = cross_validate_by_match(build_logistic_model, train_set)
    print("Cross-validated the logistic regression")

    combinations = [dict(zip(GRID, values)) for values in product(*GRID.values())]
    fold_losses = {}
    for settings in combinations:
        key = tuple(settings.values())
        fold_losses[key] = cross_validate_by_match(
            lambda s=settings: build_boosting_model(max_iter=MAX_TREES, **s), train_set
        )
        print(f"Cross-validated {settings}")

    use_experiment()
    references = {"default": fold_losses[tuple(DEFAULT.values())], "logistic": logistic_losses}
    log_cv_run("cv logistic regression", {"model": "LogisticRegression"}, logistic_losses)
    rows = []
    for settings in combinations:
        losses = fold_losses[tuple(settings.values())]
        name = "cv " + " ".join(f"{k}={v}" for k, v in settings.items())
        log_cv_run(name, {**settings, "max_iter": MAX_TREES}, losses, references)
        rows.append({
            **settings,
            "cv_log_loss": losses.mean(),
            "std": losses.std(),
            **{f"beats_{ref}": int((losses < ref_losses).sum())
               for ref, ref_losses in references.items()},
        })

    table = pd.DataFrame(rows).sort_values("cv_log_loss").reset_index(drop=True)
    print()
    print(table.round(5).to_string())
    print(f"\nLogistic regression: cv_log_loss {logistic_losses.mean():.5f}")
    print("beats_*: on how many of the 5 folds the setting did better than that reference")

    best = {k: table.loc[0, k].item() for k in GRID}
    model = train(build_boosting_model(max_iter=MAX_TREES, **best), train_set)
    scores = log_xg_run("gradient boosting (tuned)", model, train_set, test_set)
    print()
    print(f"Chosen by cross-validation: {best}")
    print("On the test matches:", {k: round(v, 4) for k, v in scores.items()})


if __name__ == "__main__":
    main()