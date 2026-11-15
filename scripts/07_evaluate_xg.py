"""Compare our two xG models with two references, and draw their calibration.

Models: a logistic regression and gradient boosting, trained on the same
matches and scored on the same test matches.
References: a model that gives every shot the average goal rate of the
training set (it knows nothing about the shot), and StatsBomb's own xG (a
professional model that also sees where the keeper and defenders stand).

Usage:
    uv run python scripts/07_evaluate_xg.py
Output:
    a score table in the terminal, docs/figures/xg_calibration.png
"""

from pathlib import Path

import duckdb
import pandas as pd

from football_ai.plots import draw_calibration
from football_ai.xg import (
    build_boosting_model,
    build_logistic_model,
    evaluate,
    load_shots,
    predict_xg,
    split_by_match,
    train,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "football.duckdb"
FIGURE_PATH = PROJECT_ROOT / "docs" / "figures" / "xg_calibration.png"


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        shots = load_shots(con)

    train_set, test_set = split_by_match(shots)
    logistic = train(build_logistic_model(), train_set)
    boosting = train(build_boosting_model(), train_set)
    test_set = test_set.assign(
        logistic_xg=predict_xg(logistic, test_set),
        boosting_xg=predict_xg(boosting, test_set),
        # Knows nothing about the shot: same value for everyone
        baseline=train_set["is_goal"].mean(),
    )

    scores = pd.DataFrame({
        "baseline (average rate)": evaluate(test_set["is_goal"], test_set["baseline"]),
        "logistic regression": evaluate(test_set["is_goal"], test_set["logistic_xg"]),
        "gradient boosting": evaluate(test_set["is_goal"], test_set["boosting_xg"]),
        "StatsBomb xG": evaluate(test_set["is_goal"], test_set["statsbomb_xg"]),
    }).T
    print(f"Test set: {len(test_set):,} shots from {test_set['match_id'].nunique()} matches")
    # Early stopping decides this, not us
    print(f"Gradient boosting stopped after {boosting.named_steps['model'].n_iter_} trees")
    print()
    print(scores.round(4).to_string())
    print()
    draw_calibration(
        test_set["is_goal"],
        {
            "Logistic regression": (test_set["logistic_xg"], "#2a78d6"),
            "Gradient boosting": (test_set["boosting_xg"], "#1f9e74"),
            "StatsBomb xG": (test_set["statsbomb_xg"], "#eb6834"),
        },
        FIGURE_PATH,
    )
    print(f"Saved {FIGURE_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()