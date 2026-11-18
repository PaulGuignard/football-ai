"""The interactive pitch as a Streamlit component (the drawing is in pitch.js)."""

from collections.abc import Callable
from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent

# Registered once: the page script runs again after every edit
_pitch = st.components.v2.component(
    "xg_pitch",
    js=(HERE / "pitch.js").read_text(),
    css=(HERE / "pitch.css").read_text(),
)


def pitch(shooter: dict, players: list[dict], board: dict, hint: str,
          on_edit: Callable[[], None], free_kick: bool = False, header: bool = False,
          key: str = "pitch") -> None:
    """Show the pitch. on_edit runs when a shirt is dropped, before the page reruns.

    board holds the text of the screen above the pitch: value, words, facts,
    note (a warning, can be empty) and broken (true when the xG couldn't be
    computed). The edit itself is read in on_edit from
    st.session_state[key].edit. With free_kick the shooter can't be dragged
    into the box, with header no further than 30 yards from goal.
    """
    _pitch(
        data={"shooter": shooter, "players": players, "board": board, "hint": hint,
              "freeKick": free_kick, "header": header},
        key=key,
        on_edit_change=on_edit,
    )