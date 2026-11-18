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
    allowed_spot,
    apply_edit,
    build_request,
    can_be_one_on_one,
    consistent_shot,
    describe,
    rarity_note,
)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
FLAGS = {
    "is_header": ("Header", "The shot is taken with the head, from 30 yards at most."),
    "is_free_kick": ("Direct free kick",
                     "Shot straight from a free kick, from outside the box."),
    "is_first_time": ("First time", "Hit without controlling the ball first."),
    "is_one_on_one": ("One-on-one", "Alone in front of the keeper."),
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
            st.session_state.shooter, st.session_state.players, edit,
            free_kick=st.session_state.get("is_free_kick", False),
            header=st.session_state.get("is_header", False))


def tidy_options() -> None:
    """Untick the options that can't go with the ones just ticked."""
    current = {flag: st.session_state[flag] for flag in FLAGS}
    current["play_pattern"] = st.session_state.play_pattern
    shot = consistent_shot(current)
    st.session_state.update(shot)
    # Ticked with the shooter somewhere he can't be: he moves to the nearest spot he can
    st.session_state.shooter = allowed_spot(
        st.session_state.shooter, shot["is_free_kick"], shot["is_header"])


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
    return {"value": f"{result['xg']:.2f}", "words": describe(result["xg"]), "facts": facts,
            "note": rarity_note(inputs["distance"])}


# Streamlit runs this whole script again after every change, so the picture
# lives in the session state to survive from one run to the next
if "shooter" not in st.session_state:
    start_again()

# A one-on-one needs nobody in the way. Unticked here, before the box is
# drawn, because Streamlit won't let a widget's value change once it's on the page
alone = can_be_one_on_one(st.session_state.shooter, st.session_state.players)
if not alone:
    st.session_state.is_one_on_one = False

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
        if flag == "is_one_on_one" and not alone:
            blocked = True
            help_text = "Never with a defender between the ball and the goal."
        shot[flag] = st.checkbox(label, key=flag, help=help_text, disabled=blocked,
                                 on_change=tidy_options)
    shot["play_pattern"] = st.selectbox(
        "How the attack started", PLAY_PATTERNS, key="play_pattern", disabled=free_kick,
        help="A header can still come from a corner or a free kick taken elsewhere.",
        on_change=tidy_options)
    shot = consistent_shot(shot, one_on_one_possible=alone)

    st.button("Start again", on_click=start_again)
    st.caption(
        "The model is gradient boosting trained on 37,481 shots from the 2015/16 Premier "
        "League, La Liga, Serie A and Ligue 1 (StatsBomb open data). It knows where the "
        "shot is taken from, where the keeper and the defenders stand and the options above. "
        "Whether the shooter is under pressure isn't asked: once the model knows where the "
        "defenders stand, it moves the xG by 0.003 on average."
    )

with left:
    pitch(
        st.session_state.shooter,
        st.session_state.players,
        xg_board(shot),
        hint="Bring a player on from the bench. Drag one off the pitch to take him off. "
             "The last defender stays: there is always one somewhere in a real match.",
        on_edit=on_edit,
        free_kick=shot["is_free_kick"],
        header=shot["is_header"],
    )