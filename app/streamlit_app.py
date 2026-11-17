"""The xG demo: move the shooter and the players with the mouse, get the xG from the API.

Run the API first (or use docker compose), then:
    uv run streamlit run app/streamlit_app.py
The API address comes from the API_URL variable, http://127.0.0.1:8000 by default.
"""

import os

import requests
import streamlit as st
from pitch import pitch  # app/pitch.py, next to this page

from football_ai.demo import (
    DEFAULT_PLAYERS,
    DEFAULT_SHOOTER,
    PLAY_PATTERNS,
    apply_edit,
    build_request,
    consistent_shot,
    describe,
)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
FLAGS = {
    "is_header": ("Header", "The shot is taken with the head."),
    "is_free_kick": ("Direct free kick", "Shot straight from a free kick, the ball still."),
    "is_first_time": ("First time", "Hit without controlling the ball first."),
    "is_one_on_one": ("One-on-one", "Alone in front of the keeper."),
    "is_under_pressure": ("Under pressure", "An opponent is closing the shooter down."),
}

st.set_page_config(page_title="How likely is this shot to go in?", page_icon="⚽",
                   layout="wide")


def start_again() -> None:
    """Back to the shot from the penalty spot."""
    st.session_state.shooter = dict(DEFAULT_SHOOTER)
    st.session_state.players = [dict(p) for p in DEFAULT_PLAYERS]


def on_edit() -> None:
    """A shirt was dropped on the pitch: apply that edit before the page reruns."""
    edit = st.session_state.pitch.edit
    if edit:
        st.session_state.shooter, st.session_state.players = apply_edit(
            st.session_state.shooter, st.session_state.players, edit)


def tidy_options() -> None:
    """Untick the options that can't go with the ones just ticked."""
    current = {flag: st.session_state[flag] for flag in FLAGS}
    current["play_pattern"] = st.session_state.play_pattern
    st.session_state.update(consistent_shot(current))


def xg_board(shot: dict) -> dict:
    """Ask the API for the xG and put the answer in the words of the screen."""
    body = build_request(st.session_state.shooter, st.session_state.players, **shot)
    try:
        response = requests.post(f"{API_URL}/xg", json=body, timeout=5)
    except requests.ConnectionError:
        return {"value": "?", "words": "The model isn't answering.",
                "facts": f"Nothing at {API_URL}. Start the API and try again.", "broken": True}
    if response.status_code != 200:
        return {"value": "?", "words": "The model refused this shot.",
                "facts": str(response.json().get("detail"))[:160], "broken": True}

    result = response.json()
    inputs = result["inputs"]
    blockers = int(inputs["defenders_in_cone"] or 0)
    facts = (
        f"{inputs['distance']:.0f} yards out with {inputs['angle']:.0f}° of goal to aim at "
        f"and {blockers} {'opponent' if blockers == 1 else 'opponents'} in the way."
    )
    return {"value": f"{result['xg']:.2f}", "words": describe(result["xg"]), "facts": facts}


# Streamlit runs this whole script again after every change, so the picture
# lives in the session state to survive from one run to the next
if "shooter" not in st.session_state:
    start_again()

st.title("How likely is this shot to go in?")
st.markdown(
    "Drag the shooter, the green 9, and the players around him. Each time you let go, "
    "an expected goals model gives the chance that the shot ends up in the net."
)

left, right = st.columns([3, 2], gap="large")

with right:
    st.subheader("The shot")
    free_kick = st.session_state.get("is_free_kick", False)
    header = st.session_state.get("is_header", False)
    shot = {}
    for flag, (label, help_text) in FLAGS.items():
        # What a direct free kick or a header rules out is greyed
        blocked = (free_kick and flag != "is_free_kick") or (
            header and flag == "is_first_time")
        shot[flag] = st.checkbox(label, key=flag, help=help_text, disabled=blocked,
                                 on_change=tidy_options)
    shot["play_pattern"] = st.selectbox(
        "How the attack started", PLAY_PATTERNS, key="play_pattern", disabled=free_kick,
        help="A header can still come from a corner or a free kick taken elsewhere.",
        on_change=tidy_options)
    shot = consistent_shot(shot)

    st.button("Start again", on_click=start_again)
    st.caption(
        "The model is gradient boosting trained on 37,481 shots from the 2015/16 Premier "
        "League, La Liga, Serie A and Ligue 1 (StatsBomb open data). It knows where the "
        "shot is taken from, where the keeper and the defenders stand and the options above."
    )

with left:
    pitch(
        st.session_state.shooter,
        st.session_state.players,
        xg_board(shot),
        hint="Bring a player on from the bench. Drag one off the pitch to take him off.",
        on_edit=on_edit,
    )