"""Stop the xG from going up when the shot moves further away.

The demo showed a direct free kick worth 0.05 at 30 yards and 0.10 at 54.
The model learnt that from a handful of shots: 6 of the 29 direct free kicks
taken from 40 to 50 yards went in. Here the v2 boosting is trained again
with a monotonic constraint (further out never helps, a wider goal never
hurts, everything else equal) and compared with the unconstrained one: on
the same 5 folds, then once on the test matches, then along the centre of
the pitch to see the curves the two models draw.

Usage:
    uv run python scripts/16_monotonic_xg.py
Output:
    two score tables in the terminal, docs/figures/xg_by_distance.png
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from football_ai.plots import BACKGROUND, GREY
from football_ai.shot_inputs import shot_row
from football_ai.tracking import log_cv_run, log_xg_run, use_experiment
from football_ai.xg import (
    build_boosting_model,
    cross_validate_by_match,
    evaluate,
    load_shots,
    predict_xg,
    split_by_match,
    train,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "football.duckdb"
FIGURE_PATH = PROJECT_ROOT / "docs" / "figures" / "xg_by_distance.png"

# The v2 boosting kept in 12_compare_feature_sets.py
BOOSTING = {"learning_rate": 0.05, "max_leaf_nodes": 4, "min_samples_leaf": 20, "max_iter": 1000}
KEEPER_ON_HIS_LINE = {"x": 119.0, "y": 40.0, "teammate": False, "keeper": True}
# Three kinds of shot, all from the middle of the pitch with the keeper on his
# line and a defender 4 yards to the side. Never nobody at all: only 1 of the
# 37,481 shots had no defender in the picture, so the model knows nothing there
SHOT_KINDS = {
    "Shot with the foot": {},
    "Header": {"is_header": True},
    "Direct free kick": {"is_free_kick": True, "play_pattern": "From Free Kick"},
}
DISTANCES = np.arange(6, 61, 1.0)


def beside(x: float) -> dict:
    """A defender level with the shooter, 4 yards to his left: close, but not in the way."""
    return {"x": x, "y": 44.0, "teammate": False, "keeper": False}


def xg_along_the_centre(model, kind: dict) -> np.ndarray:
    """xG of the same shot taken from further and further out, straight in front of goal."""
    rows = pd.concat([
        shot_row(120 - d, 40.0, [KEEPER_ON_HIS_LINE, beside(120 - d)], **kind) for d in DISTANCES
    ])
    return predict_xg(model, rows).to_numpy()


def furthest_seen(shots: pd.DataFrame, kind: dict) -> float:
    """The longest distance a shot of this kind was taken from in the training data."""
    if kind.get("is_header"):
        return shots.loc[shots["is_header"], "distance"].max()
    if kind.get("is_free_kick"):
        return shots.loc[shots["is_free_kick"], "distance"].max()
    return shots.loc[~shots["is_header"] & ~shots["is_free_kick"], "distance"].max()


def draw_curves(free, constrained, train_set: pd.DataFrame, path: Path) -> None:
    """One panel per kind of shot: xG against distance, before and after the constraint."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    fig.set_facecolor(BACKGROUND)
    for ax, (title, kind) in zip(axes, SHOT_KINDS.items(), strict=True):
        ax.set_facecolor(BACKGROUND)
        ax.plot(DISTANCES, xg_along_the_centre(free, kind), color=GREY, linestyle="--",
                linewidth=1.8, label="without the constraint")
        ax.plot(DISTANCES, xg_along_the_centre(constrained, kind), color="#1f9e74",
                linewidth=2.2, label="with the constraint")
        furthest = furthest_seen(train_set, kind)
        if furthest < DISTANCES.max():
            # Beyond this, the model is guessing: no such shot in the data
            ax.axvspan(furthest, DISTANCES.max(), color=GREY, alpha=0.12, linewidth=0)
            ax.text(furthest + 1, 0.42, "never seen\nin the data", color=GREY, fontsize=8,
                    va="top")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Distance to goal (yards)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("xG, keeper on his line, a defender beside")
    axes[0].legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        shots = load_shots(con, freeze_frames=True)
    train_set, test_set = split_by_match(shots)

    def build_free():
        return build_boosting_model("v2", **BOOSTING)

    def build_constrained():
        return build_boosting_model("v2", monotonic=True, **BOOSTING)

    free_losses = cross_validate_by_match(build_free, train_set)
    constrained_losses = cross_validate_by_match(build_constrained, train_set)
    use_experiment()
    log_cv_run("cv boosting v2, monotonic", {**BOOSTING, "monotonic": True}, constrained_losses,
               {"unconstrained": free_losses}, feature_set="v2")

    print("Cross-validation on the training matches (log loss, 5 folds):")
    print(pd.DataFrame({
        "without": free_losses, "with the constraint": constrained_losses,
        "difference": constrained_losses - free_losses,
    }, index=[f"fold {i + 1}" for i in range(5)]).round(5).to_string())
    print(f"mean difference {np.mean(constrained_losses - free_losses):+.5f}, "
          f"constraint better on {int((constrained_losses < free_losses).sum())} of 5 folds")

    free = train(build_free(), train_set)
    constrained = train(build_constrained(), train_set)
    baseline = pd.Series(train_set["is_goal"].mean(), index=test_set.index)
    scores = {
        "baseline (average rate)": evaluate(test_set["is_goal"], baseline),
        "boosting v2": evaluate(test_set["is_goal"], predict_xg(free, test_set)),
        # The unconstrained one was already logged by 12_compare_feature_sets.py
        "boosting v2, monotonic": log_xg_run("boosting v2, monotonic", constrained,
                                             train_set, test_set, feature_set="v2"),
        "StatsBomb xG": evaluate(test_set["is_goal"], test_set["statsbomb_xg"]),
    }
    table = pd.DataFrame(scores).T
    gap = table.loc["baseline (average rate)", "log_loss"] - table.loc["StatsBomb xG", "log_loss"]
    table["gap_closed"] = (table.loc["baseline (average rate)", "log_loss"] - table["log_loss"]) / gap
    print(f"\nOn the test matches ({len(test_set):,} shots):")
    print(table.round(4).to_string())

    draw_curves(free, constrained, train_set, FIGURE_PATH)
    print(f"\nSaved {FIGURE_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()