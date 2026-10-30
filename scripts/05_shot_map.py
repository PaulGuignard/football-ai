"""Draw where shots are taken from and how often each zone scores.

Two half-pitches side by side: the number of shots per zone, then the share
of those shots that end up in the net. Same shots as the xG model (no
penalties, no shots straight from a corner).

Usage:
    uv run python scripts/05_shot_map.py
Output:
    docs/figures/shot_map.png
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import VerticalPitch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "football.duckdb"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "figures" / "shot_map.png"

# Below this many shots, a zone's conversion rate is mostly noise: leave it blank
MIN_SHOTS_PER_ZONE = 30

# One hue per panel, light = little, dark = a lot
BLUES = LinearSegmentedColormap.from_list("blues", ["#eaf2fd", "#86b6ef", "#2a78d6", "#0d366b"])
ORANGES = LinearSegmentedColormap.from_list("oranges", ["#fdeee6", "#f3a17f", "#eb6834", "#8a3212"])

QUERY = """
SELECT x, y, shot_outcome = 'Goal' AS is_goal
FROM events
WHERE type = 'Shot' AND shot_type NOT IN ('Penalty', 'Corner')
"""


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        shots = con.sql(QUERY).df()

    pitch = VerticalPitch(
        pitch_type="statsbomb", half=True, pitch_color="#fcfcfb", line_color="#8a8a86",
        linewidth=1, line_zorder=2, pad_bottom=-10,
    )
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(12, 7))
    fig.set_facecolor("#fcfcfb")

    # 4-yard zones: fine enough to see the six-yard box, big enough to hold shots
    bins = (30, 20)
    counts = pitch.bin_statistic(shots["x"], shots["y"], statistic="count", bins=bins)
    rates = pitch.bin_statistic(
        shots["x"], shots["y"], values=shots["is_goal"].astype(float), statistic="mean", bins=bins
    )
    rates["statistic"] = np.where(counts["statistic"] >= MIN_SHOTS_PER_ZONE, rates["statistic"], np.nan)
    counts["statistic"] = np.where(counts["statistic"] > 0, counts["statistic"], np.nan)

    left = pitch.heatmap(counts, ax=axes[0], cmap=BLUES, edgecolor="#fcfcfb", linewidth=1)
    right = pitch.heatmap(
        rates, ax=axes[1], cmap=ORANGES, vmin=0, vmax=0.6, edgecolor="#fcfcfb", linewidth=1
    )

    for ax, mesh, title, label, fmt in [
        (axes[0], left, "Where shots are taken", "Shots per zone", "{x:,.0f}"),
        (axes[1], right, "How often they score", "Share of shots that score", "{x:.0%}"),
    ]:
        ax.set_title(title, fontsize=14, color="#1f1f1d", pad=8)
        bar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02)
        bar.set_label(label, color="#5f5f5b")
        bar.outline.set_visible(False)
        bar.ax.tick_params(colors="#5f5f5b", length=0)
        bar.ax.xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter(fmt))

    n_shots, n_goals = len(shots), int(shots["is_goal"].sum())
    fig.text(
        0.5, -0.03,
        f"Premier League, La Liga, Serie A and Ligue 1, 2015/16. {n_shots:,} shots, "
        f"{n_goals:,} goals (no penalties, no direct corners). Zones with fewer than "
        f"{MIN_SHOTS_PER_ZONE} shots left blank. Data: StatsBomb open data.",
        ha="center", fontsize=9, color="#5f5f5b",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()