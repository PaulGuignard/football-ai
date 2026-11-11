"""Train the first xG model (logistic regression) and look at a few predictions.

Needs the shot_features table: run `uv run python -m football_ai.features`
first if it isn't built yet.

Usage:
    uv run python scripts/06_train_xg.py
"""

from pathlib import Path

import duckdb

from football_ai.xg import (
    build_logistic_model,
    load_shots,
    predict_xg,
    split_by_match,
    train,
)

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "football.duckdb"


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        shots = load_shots(con)

    train_set, test_set = split_by_match(shots)
    print(f"Training on {len(train_set):,} shots, testing on {len(test_set):,}")

    model = train(build_logistic_model(), train_set)
    test_set = test_set.assign(xg=predict_xg(model, test_set))

    # A first sanity check: on average the model should predict about as many
    # goals as there really were
    print(f"Goals in the test set:   {test_set['is_goal'].sum():,}")
    print(f"Goals the model expects: {test_set['xg'].sum():,.0f}")
    print()

    columns = ["distance", "angle", "is_header", "play_pattern", "is_goal", "xg", "statsbomb_xg"]
    print("A few test shots, with our xG next to StatsBomb's:")
    print(test_set[columns].sample(8, random_state=1).round(2).to_string(index=False))


if __name__ == "__main__":
    main()