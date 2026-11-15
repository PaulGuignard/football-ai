"""Figures shared by several scripts."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve

BACKGROUND = "#fcfcfb"
GREY = "#8a8a86"


def draw_calibration(
    is_goal: pd.Series,
    curves: dict[str, tuple[pd.Series, str]],
    path: Path,
    title: str = "Calibration on the test matches",
) -> None:
    """For shots grouped by predicted xG, compare the prediction to reality.

    curves maps a label to (xg of each shot, colour). Each curve has 10 groups
    with the same number of shots. A well calibrated model sits on the
    diagonal: shots given 20% go in about 20% of the time.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    top = 0.5
    for label, (xg, color) in curves.items():
        real_rate, predicted = calibration_curve(is_goal, xg, n_bins=10, strategy="quantile")
        ax.plot(predicted, real_rate, marker="o", markersize=5, linewidth=2,
                color=color, label=label)
        top = max(top, predicted.max(), real_rate.max())
    # Leave room for the best chances, which go higher once the keeper is known
    top = math.ceil(top * 10) / 10

    ax.plot([0, top], [0, top], linestyle="--", color=GREY, linewidth=1)
    ax.text(0.6 * top, 0.54 * top, "perfect calibration", color=GREY, rotation=45, fontsize=9)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Predicted chance of scoring (average xG of the group)")
    ax.set_ylabel("Share of those shots that were goals")
    ax.set_title(title, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)