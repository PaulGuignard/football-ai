"""Show what the first xG model learnt, input by input.

Each odds ratio says by how much the odds of scoring are multiplied by an
input, everything else being equal. That's what the exploration could not
do: separate the effect of, say, pressure from the effect of distance.

Usage:
    uv run python scripts/08_explain_xg.py
"""

from pathlib import Path

import duckdb

from football_ai.xg import (
    build_logistic_model,
    coefficients,
    load_shots,
    split_by_match,
    train,
)

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "football.duckdb"


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        shots = load_shots(con)

    train_set, _ = split_by_match(shots)
    model = train(build_logistic_model(), train_set)

    print("Distance and angle: effect of one standard deviation")
    print(
        f"  (distance: {train_set['distance'].std():.1f} yards, "
        f"angle: {train_set['angle'].std():.1f} degrees)"
    )
    print("Flags: effect of 'yes' compared with 'no'")
    print("How the attack started: compare the values with each other")
    print()
    print(coefficients(model).round(2).to_string(index=False))


if __name__ == "__main__":
    main()