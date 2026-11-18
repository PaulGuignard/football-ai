"""Train the final xG model on every shot and save it for the API.

The evaluation is over (see the project log), so the model now learns from
all 37,481 shots, test matches included. The scores to quote stay the ones
measured on the test matches. The file is small and goes into Git, so that
Docker and the online demo get it without needing the database.

Usage:
    uv run python scripts/15_export_model.py
Output:
    models/xg_model.skops and models/xg_model.json
"""

from pathlib import Path

import duckdb

from football_ai.model_store import MODEL_PATH, load_model, save_model
from football_ai.shot_inputs import shot_row
from football_ai.xg import build_boosting_model, load_shots, predict_xg, train

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "football.duckdb"

# The v2 boosting kept in 12_compare_feature_sets.py, with the constraint
# from 16_monotonic_xg.py: further out never raises the xG
BOOSTING = {"learning_rate": 0.05, "max_leaf_nodes": 4, "min_samples_leaf": 20, "max_iter": 1000}

KEEPER_ON_HIS_LINE = {"x": 119.0, "y": 40.0, "teammate": False, "keeper": True}
# A shot always has some defender around: only 1 of the 37,481 had none
DEFENDER_ON_THE_EDGE = {"x": 112.0, "y": 47.0, "teammate": False, "keeper": False}


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        shots = load_shots(con, freeze_frames=True)

    model = train(build_boosting_model("v2", monotonic=True, **BOOSTING), shots)
    save_model(model, model_name="gradient boosting v2, monotonic",
               settings={**BOOSTING, "monotonic": True}, shots=len(shots))
    print(f"Trained on {len(shots):,} shots, saved {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"File size: {MODEL_PATH.stat().st_size / 1024:.0f} KB")

    # Reload the file and check it gives exactly what the model in memory gives
    reloaded = load_model()
    assert predict_xg(reloaded, shots).equals(predict_xg(model, shots))

    print("\nA few shots with the keeper on his line and a defender off to the side:")
    for label, x, y, header in [
        ("penalty spot, foot", 108.0, 40.0, False),
        ("penalty spot, header", 108.0, 40.0, True),
        ("edge of the box", 102.0, 40.0, False),
        ("tight angle", 114.0, 66.0, False),
        ("50 yards out, foot", 70.0, 40.0, False),
    ]:
        row = shot_row(x, y, [KEEPER_ON_HIS_LINE, DEFENDER_ON_THE_EDGE], is_header=header)
        print(f"  {label:<22} xG {predict_xg(reloaded, row).iloc[0]:.3f}")


if __name__ == "__main__":
    main()