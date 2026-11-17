"""What the Streamlit demo needs besides its page: the picture, the edits, the rules.

Kept apart from the page so it can be tested, and kept light: no model and
no scikit-learn here, since the demo only talks to the API. The pitch itself
is drawn in the browser (app/pitch), which sends back one edit each time a
shirt is dropped. This module applies those edits and keeps the shot options
consistent with each other.
"""

# The ways an attack can start, as StatsBomb names them (same list as the API)
PLAY_PATTERNS = [
    "Regular Play", "From Free Kick", "From Throw In", "From Corner", "From Counter",
    "From Goal Kick", "From Keeper", "From Kick Off", "Other",
]

# A starting picture: a shot from the penalty spot, the keeper on his line, a
# defender in the way and a teammate on the side
DEFAULT_SHOOTER = {"x": 108.0, "y": 40.0}
DEFAULT_PLAYERS = [
    {"x": 119.0, "y": 40.0, "teammate": False, "keeper": True},
    {"x": 113.0, "y": 40.5, "teammate": False, "keeper": False},
    {"x": 110.0, "y": 46.0, "teammate": True, "keeper": False},
]
# A freeze frame never shows more than the 21 other players
MAX_PLAYERS = 21

# Who can be brought on from the bench under the pitch
NEWCOMERS = {
    "defender": {"teammate": False, "keeper": False},
    "teammate": {"teammate": True, "keeper": False},
    "keeper": {"teammate": False, "keeper": True},
}

# In the 37,481 shots, a direct free kick is never a header, never first time
# and never under pressure, and it always comes from a free kick
FREE_KICK_EXCLUDES = ("is_header", "is_first_time", "is_one_on_one", "is_under_pressure")


def on_pitch(x: float, y: float) -> bool:
    """True if the spot is on the half of the pitch the demo shows."""
    return 60 <= x <= 120 and 0 <= y <= 80


def is_their_keeper(player: dict) -> bool:
    """The keeper of the defending side, the only one the model looks at."""
    return player["keeper"] and not player["teammate"]


def apply_edit(shooter: dict, players: list[dict], edit: dict) -> tuple[dict, list[dict]]:
    """The new shooter and players after one edit made on the pitch.

    An edit is one of
      {"action": "move", "who": "shooter" or a player's index, "x": .., "y": ..}
      {"action": "add", "kind": "defender" | "teammate" | "keeper", "x": .., "y": ..}
      {"action": "remove", "who": a player's index}
    Nothing is changed in place. Anything that doesn't make sense (a spot off
    the pitch, an index that no longer exists) leaves the picture as it was.
    """
    players = [dict(p) for p in players]
    action = edit.get("action")
    who = edit.get("who")
    spot = (float(edit.get("x", -1)), float(edit.get("y", -1)))

    if action == "move" and on_pitch(*spot):
        if who == "shooter":
            return {"x": spot[0], "y": spot[1]}, players
        if isinstance(who, int) and 0 <= who < len(players):
            players[who].update(x=spot[0], y=spot[1])
    elif action == "add" and edit.get("kind") in NEWCOMERS and on_pitch(*spot):
        newcomer = {"x": spot[0], "y": spot[1], **NEWCOMERS[edit["kind"]]}
        if is_their_keeper(newcomer):
            # Only one keeper in goal: the new one takes the old one's place
            players = [p for p in players if not is_their_keeper(p)]
        if len(players) < MAX_PLAYERS:
            players.append(newcomer)
    elif action == "remove" and isinstance(who, int) and 0 <= who < len(players):
        del players[who]
    return dict(shooter), players


def consistent_shot(shot: dict) -> dict:
    """The shot options with the impossible combinations taken out.

    A direct free kick switches off the flags it never comes with and sets
    how the attack started. A header is almost never tagged first time in the
    data (5 headers out of 6,115), so that flag goes off too.
    """
    shot = dict(shot)
    if shot.get("is_free_kick"):
        shot.update(dict.fromkeys(FREE_KICK_EXCLUDES, False), play_pattern="From Free Kick")
    if shot.get("is_header"):
        shot["is_first_time"] = False
    return shot


def build_request(shooter: dict, players: list[dict], **shot) -> dict:
    """The JSON body for POST /xg."""
    return {
        "x": float(shooter["x"]),
        "y": float(shooter["y"]),
        **shot,
        "players": [
            {"x": float(p["x"]), "y": float(p["y"]),
             "teammate": bool(p["teammate"]), "keeper": bool(p["keeper"])}
            for p in players
        ],
    }


def describe(xg: float) -> str:
    """The xG put in words a non-specialist can picture."""
    in_100 = round(xg * 100)
    if in_100 < 1:
        return "Fewer than 1 shot in 100 like this one goes in."
    if in_100 == 1:
        return "About 1 shot in 100 like this one goes in."
    return f"About {in_100} shots in 100 like this one go in."