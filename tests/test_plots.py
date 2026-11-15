"""Tests for the shared figures."""

import numpy as np
import pandas as pd

from football_ai.plots import draw_calibration


def test_calibration_figure_is_written(tmp_path):
    rng = np.random.default_rng(0)
    xg = pd.Series(rng.uniform(0, 0.6, 500))
    # Goals drawn with exactly those chances: a well calibrated "model"
    is_goal = pd.Series((rng.random(500) < xg).astype(int))
    path = tmp_path / "figures" / "calibration.png"

    draw_calibration(is_goal, {"Test model": (xg, "#2a78d6")}, path)

    assert path.exists() and path.stat().st_size > 0