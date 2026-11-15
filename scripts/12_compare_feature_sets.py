"""Do the freeze frames help? Compare v1 and v2 inputs by cross-validation, then test once.

v1 is the eight inputs from the event data, v2 adds the four from the freeze
frames. Every candidate is scored on the same 5 folds of the training
matches and logged in MLflow. The boosting's tree size is chosen again for v2,
since more inputs may call for bigger trees. Then the v2 models are scored
once on the test matches, next to the v1 ones and StatsBomb's xG.

Usage:
    uv run python scripts/12_compare_feature_sets.py
"""

from pathlib import Path

import duckdb
import pandas as pd

from football_ai.tracking import log_cv_run, log_xg_run, use_experiment
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

# What won in 10_tune_boosting.py; only the tree size is searched again
BOOSTING = {"learning_rate": 0.05, "min_samples_leaf": 20, "max_iter": 1000}
TREE_SIZES = [4, 7, 15, 31]


def candidates() -> dict[str, tuple[str, dict, object]]:
    """Every model to cross-validate: name -> (feature set, settings, builder)."""
    models = {
        "logistic v1": ("v1", {"model": "LogisticRegression"},
                        lambda: build_logistic_model("v1")),
        "logistic v2": ("v2", {"model": "LogisticRegression"},
                        lambda: build_logistic_model("v2")),
        "boosting v1, 4 leaves": ("v1", {**BOOSTING, "max_leaf_nodes": 4},
                                  lambda: build_boosting_model("v1", max_leaf_nodes=4, **BOOSTING)),
    }
    for leaves in TREE_SIZES:
        models[f"boosting v2, {leaves} leaves"] = (
            "v2",
            {**BOOSTING, "max_leaf_nodes": leaves},
            lambda n=leaves: build_boosting_model("v2", max_leaf_nodes=n, **BOOSTING),
        )
    return models


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        shots = load_shots(con, freeze_frames=True)
    train_set, test_set = split_by_match(shots)

    models = candidates()
    fold_losses = {}
    for name, (_, _, build) in models.items():
        fold_losses[name] = cross_validate_by_match(build, train_set)
        print(f"Cross-validated {name}")

    # Same model, v1 against v2: does the extra information pay off? And in
    # v2, does the boosting beat the logistic regression fold by fold?
    references = {
        "logistic_v1": fold_losses["logistic v1"],
        "boosting_v1": fold_losses["boosting v1, 4 leaves"],
        "logistic_v2": fold_losses["logistic v2"],
    }
    use_experiment()
    rows = []
    for name, (feature_set, settings, _) in models.items():
        losses = fold_losses[name]
        log_cv_run(f"cv {name}", settings, losses, references, feature_set=feature_set)
        rows.append({
            "model": name,
            "cv_log_loss": losses.mean(),
            "std": losses.std(),
            **{f"beats_{ref}": int((losses < ref_losses).sum())
               for ref, ref_losses in references.items()},
        })
    print()
    print(pd.DataFrame(rows).sort_values("cv_log_loss").round(5).to_string(index=False))
    print("beats_*: on how many of the 5 folds the model did better than that reference")

    # The v2 boosting with the best cross-validated tree size goes to the test
    best_leaves = min(TREE_SIZES, key=lambda n: fold_losses[f"boosting v2, {n} leaves"].mean())
    finalists = {
        "logistic v1": models["logistic v1"],
        "boosting v1, 4 leaves": models["boosting v1, 4 leaves"],
        "logistic v2": models["logistic v2"],
        f"boosting v2, {best_leaves} leaves": models[f"boosting v2, {best_leaves} leaves"],
    }
    test_scores = {}
    for name, (feature_set, _, build) in finalists.items():
        model = train(build(), train_set)
        if feature_set == "v2":
            # The v1 test runs were already logged by 09 and 10
            test_scores[name] = log_xg_run(name, model, train_set, test_set, feature_set="v2")
        else:
            test_scores[name] = evaluate(test_set["is_goal"], predict_xg(model, test_set))
    test_scores["StatsBomb xG"] = evaluate(test_set["is_goal"], test_set["statsbomb_xg"])

    print(f"\nOn the test matches ({len(test_set):,} shots, used once):")
    print(pd.DataFrame(test_scores).T.round(4).to_string())


if __name__ == "__main__":
    main()