"""Raw positions against inputs built by hand: the Deep Sets network on trial.

The network reads every visible player as they are, plus the eight shot
inputs of v1. It is scored on the same 5 folds of the training matches as
the logistic regression and the boosting with the hand-built inputs (v2),
then once on the test matches, next to StatsBomb's xG. Its calibration is
drawn next to the boosting's.

Usage:
    uv run python scripts/13_compare_deep_sets.py
Output:
    two score tables in the terminal, docs/figures/xg_calibration_deep_sets.png
"""

from pathlib import Path

import duckdb
import pandas as pd

from football_ai.deep_sets import (
    cross_validate_deep_sets,
    load_players,
    player_arrays,
    train_deep_sets,
)
from football_ai.plots import draw_calibration
from football_ai.tracking import log_cv_run, log_deep_sets_run, use_experiment
from football_ai.xg import (
    build_boosting_model,
    build_logistic_model,
    cross_validate_by_match,
    evaluate,
    load_shots,
    predict_xg,
    split_by_match,
    train,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "football.duckdb"
FIGURE_PATH = PROJECT_ROOT / "docs" / "figures" / "xg_calibration_deep_sets.png"

# The v2 boosting kept in 12_compare_feature_sets.py
BOOSTING = {"learning_rate": 0.05, "max_leaf_nodes": 4, "min_samples_leaf": 20, "max_iter": 1000}
# The network's own settings, the defaults of train_deep_sets
DEEP_SETS = {"hidden": 32, "learning_rate": 1e-3, "batch_size": 256, "patience": 10}


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        shots = load_shots(con, freeze_frames=True)
        players = load_players(con)
    train_set, test_set = split_by_match(shots)
    # Row i of the arrays is row i of the shots: always build them from the same table
    train_positions, train_mask = player_arrays(train_set, players)
    test_positions, test_mask = player_arrays(test_set, players)

    fold_losses = {
        "logistic v2": cross_validate_by_match(lambda: build_logistic_model("v2"), train_set),
        "boosting v2": cross_validate_by_match(
            lambda: build_boosting_model("v2", **BOOSTING), train_set
        ),
        "deep sets": cross_validate_deep_sets(train_set, train_positions, train_mask, **DEEP_SETS),
    }
    print("Cross-validated the three models")

    references = {"logistic_v2": fold_losses["logistic v2"], "boosting_v2": fold_losses["boosting v2"]}
    use_experiment()
    log_cv_run("cv deep sets", {"model": "DeepSets", **DEEP_SETS}, fold_losses["deep sets"],
               references, feature_set="raw positions")
    rows = [
        {
            "model": name,
            "cv_log_loss": losses.mean(),
            "std": losses.std(),
            **{f"beats_{ref}": int((losses < ref_losses).sum())
               for ref, ref_losses in references.items()},
        }
        for name, losses in fold_losses.items()
    ]
    print()
    print(pd.DataFrame(rows).sort_values("cv_log_loss").round(5).to_string(index=False))
    print("beats_*: on how many of the 5 folds the model did better than that reference")

    # Once on the test matches. The boosting's test run was logged by 12
    boosting = train(build_boosting_model("v2", **BOOSTING), train_set)
    deep_sets = train_deep_sets(train_set, train_positions, train_mask, **DEEP_SETS)
    xg = {
        "boosting v2": predict_xg(boosting, test_set),
        "deep sets": deep_sets.predict_xg(test_set, test_positions, test_mask),
    }
    test_scores = {
        "boosting v2": evaluate(test_set["is_goal"], xg["boosting v2"]),
        "deep sets": log_deep_sets_run("deep sets", deep_sets, train_set, test_set,
                                       test_positions, test_mask, DEEP_SETS),
        "StatsBomb xG": evaluate(test_set["is_goal"], test_set["statsbomb_xg"]),
    }
    print(f"\nOn the test matches ({len(test_set):,} shots, used once):")
    print(pd.DataFrame(test_scores).T.round(4).to_string())
    print(f"Deep Sets kept the weights of epoch {deep_sets.best_epoch}")

    draw_calibration(
        test_set["is_goal"],
        {
            "Gradient boosting, v2": (xg["boosting v2"], "#1f9e74"),
            "Deep Sets, raw positions": (xg["deep sets"], "#7b5ea7"),
            "StatsBomb xG": (test_set["statsbomb_xg"], "#eb6834"),
        },
        FIGURE_PATH,
        title="Calibration on the test matches: raw positions",
    )
    print(f"\nSaved {FIGURE_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()