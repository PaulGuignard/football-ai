"""Tests for flattening shot freeze frames, on hand-made events."""

import pandas as pd

from football_ai.freeze_frames import COLUMNS, explode_freeze_frames


def player(name, x, y, teammate, position="Center Back"):
    """One player as StatsBomb writes it inside a freeze frame."""
    return {
        "location": [x, y],
        "player": {"id": 1, "name": name},
        "position": {"id": 1, "name": position},
        "teammate": teammate,
    }


def make_events():
    """A pass, a shot seen with three players, and a penalty with no freeze frame."""
    return pd.DataFrame({
        "id": ["pass-1", "shot-1", "shot-2"],
        "type": ["Pass", "Shot", "Shot"],
        "shot_freeze_frame": [
            None,
            [
                player("Keeper", 119.0, 40.0, False, position="Goalkeeper"),
                player("Defender", 110.0, 38.0, False),
                player("Striker", 112.0, 45.0, True),
            ],
            None,
        ],
    })


def test_one_row_per_player_seen_in_the_shot():
    frames = explode_freeze_frames(make_events(), match_id=7)

    assert len(frames) == 3
    assert (frames["shot_id"] == "shot-1").all()
    assert (frames["match_id"] == 7).all()


def test_locations_and_flags_are_split_out():
    frames = explode_freeze_frames(make_events(), match_id=7).set_index("player")

    assert frames.loc["Keeper", "position"] == "Goalkeeper"
    assert frames.loc["Defender", "x"] == 110.0
    assert frames.loc["Defender", "y"] == 38.0
    assert bool(frames.loc["Striker", "teammate"]) is True
    assert bool(frames.loc["Keeper", "teammate"]) is False


def test_a_match_without_any_freeze_frame_gives_an_empty_table():
    # No shot has a freeze frame, so the column isn't there at all
    events = pd.DataFrame({"id": ["pass-1"], "type": ["Pass"]})
    frames = explode_freeze_frames(events, match_id=7)

    assert frames.empty
    assert list(frames.columns) == COLUMNS