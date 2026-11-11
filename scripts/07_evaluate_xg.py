"""Evaluate the first xG model against two references, and draw its calibration.

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
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve

from football_ai.xg import (
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


def draw_calibration(test_set: pd.DataFrame) -> None:
    """For shots grouped by predicted xG, compare the prediction to reality."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#8a8a86", linewidth=1)

    for column, label, color in [
        ("xg", "Our model (logistic regression)", "#2a78d6"),
        ("statsbomb_xg", "StatsBomb xG", "#eb6834"),
    ]:
        # 10 groups with the same number of shots in each
        real_rate, predicted = calibration_curve(
            test_set["is_goal"], test_set[column], n_bins=10, strategy="quantile"
        )
        ax.plot(predicted, real_rate, marker="o", markersize=5, linewidth=2,
                color=color, label=label)

    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("Predicted chance of scoring (average xG of the group)")
    ax.set_ylabel("Share of those shots that were goals")
    ax.set_title("Calibration on the test matches", loc="left")
    ax.text(0.30, 0.27, "perfect calibration", color="#8a8a86", rotation=45, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved {FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        shots = load_shots(con)

    train_set, test_set = split_by_match(shots)
    model = train(build_logistic_model(), train_set)
    test_set = test_set.assign(
        xg=predict_xg(model, test_set),
        # Knows nothing about the shot: same value for everyone
        baseline=train_set["is_goal"].mean(),
    )

    scores = pd.DataFrame({
        "baseline (average rate)": evaluate(test_set["is_goal"], test_set["baseline"]),
        "our model": evaluate(test_set["is_goal"], test_set["xg"]),
        "StatsBomb xG": evaluate(test_set["is_goal"], test_set["statsbomb_xg"]),
    }).T
    print(f"Test set: {len(test_set):,} shots from {test_set['match_id'].nunique()} matches")
    print(scores.round(4).to_string())
    print()
    draw_calibration(test_set)


if __name__ == "__main__":
    main()