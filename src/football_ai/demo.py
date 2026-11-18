"""What the Streamlit demo needs besides its page: the picture, the edits, the rules.

Kept apart from the page so it can be tested, and kept light: no model and
no scikit-learn here, since the demo only talks to the API. The pitch itself
is drawn in the browser (app/pitch), which sends back one edit each time a
shirt is dropped. This module applies those edits and keeps the shot options
consistent with each other and with the picture.
"""

import math

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
    {"x": 112.5, "y": 40.5, "teammate": False, "keeper": False},
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
# and never a one-on-one, and it always comes from a free kick
FREE_KICK_EXCLUDES = ("is_header", "is_first_time", "is_one_on_one")
# The longest header in the data is 27 yards out; 30 leaves a little room
HEADER_REACH = 30.0
# Fewer than 1% of the shots are taken from further out than this
RARE_DISTANCE = 40.0
GOAL = (120.0, 40.0)
POSTS = (36.0, 44.0)


def on_pitch(x: float, y: float) -> bool:
    """True if the spot is on the half of the pitch the demo shows."""
    return 60 <= x <= 120 and 0 <= y <= 80


def in_box(x: float, y: float) -> bool:
    """True if the spot is inside the penalty area. The lines count as outside."""
    return x > 102 and 18 < y < 62


def keep_out_of_box(spot: dict) -> dict:
    """The spot itself, or the nearest point on the edge of the box if it's inside.

    Used for direct free kicks: none of the 1,749 in the data is taken from
    inside the box, since a foul there gives a penalty.
    """
    x, y = spot["x"], spot["y"]
    if not in_box(x, y):
        return dict(spot)
    ways_out = [
        (x - 102, {"x": 102.0, "y": y}),   # through the front
        (y - 18, {"x": x, "y": 18.0}),     # through one side
        (62 - y, {"x": x, "y": 62.0}),     # or the other
    ]
    return min(ways_out, key=lambda way: way[0])[1]


def keep_in_heading_range(spot: dict) -> dict:
    """The spot itself, or the point HEADER_REACH yards out on the way to goal if it's further.

    Nobody heads the ball in from 50 yards: there isn't the power.
    """
    distance = math.dist((spot["x"], spot["y"]), GOAL)
    if distance <= HEADER_REACH:
        return dict(spot)
    pull = HEADER_REACH / distance
    return {"x": GOAL[0] + (spot["x"] - GOAL[0]) * pull,
            "y": GOAL[1] + (spot["y"] - GOAL[1]) * pull}


def allowed_spot(spot: dict, free_kick: bool = False, header: bool = False) -> dict:
    """Where the shooter ends up given the options: out of the box, within heading range."""
    if free_kick:
        spot = keep_out_of_box(spot)
    if header:
        spot = keep_in_heading_range(spot)
    return dict(spot)


def rarity_note(distance: float) -> str:
    """A warning for shots taken from where the model has seen almost none."""
    if distance > RARE_DISTANCE:
        return "The model has seen very few shots from this far, so take this number with care."
    return ""


def is_their_keeper(player: dict) -> bool:
    """The keeper of the defending side, the only one the model looks at."""
    return player["keeper"] and not player["teammate"]


def is_defender(player: dict) -> bool:
    """An outfield player of the defending side."""
    return not player["teammate"] and not player["keeper"]


def in_the_way(player: dict, shooter: dict) -> bool:
    """Is the player inside the triangle between the ball and the two posts?

    The same geometry as in_shot_cone in shot_inputs.py, written again here
    because the demo image has neither pandas nor scikit-learn. A test checks
    that the two agree.
    """
    corners = [(shooter["x"], shooter["y"]), (GOAL[0], POSTS[0]), (GOAL[0], POSTS[1])]
    sides = []
    for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1], strict=True):
        sides.append((bx - ax) * (player["y"] - ay) - (by - ay) * (player["x"] - ax))
    return all(s >= 0 for s in sides) or all(s <= 0 for s in sides)


def can_be_one_on_one(shooter: dict, players: list[dict]) -> bool:
    """True if no defender stands between the ball and the goal.

    All 1,850 one-on-ones in the data have nobody in that triangle. The
    reverse isn't true: most shots with an empty triangle aren't one-on-ones.
    """
    return not any(is_defender(p) and in_the_way(p, shooter) for p in players)


def apply_edit(shooter: dict, players: list[dict], edit: dict,
               free_kick: bool = False, header: bool = False) -> tuple[dict, list[dict]]:
    """The new shooter and players after one edit made on the pitch.

    An edit is one of
      {"action": "move", "who": "shooter" or a player's index, "x": .., "y": ..}
      {"action": "add", "kind": "defender" | "teammate" | "keeper", "x": .., "y": ..}
      {"action": "remove", "who": a player's index}
    Nothing is changed in place. Anything that doesn't make sense (a spot off
    the pitch, an index that no longer exists) leaves the picture as it was.
    The last defender can't be taken off: only 1 of the 37,481 shots had no
    defender anywhere in the picture, so the model knows nothing about it.
    The shooter is kept where the options allow (see allowed_spot).
    """
    players = [dict(p) for p in players]
    action = edit.get("action")
    who = edit.get("who")
    spot = (float(edit.get("x", -1)), float(edit.get("y", -1)))

    if action == "move" and on_pitch(*spot):
        if who == "shooter":
            return allowed_spot({"x": spot[0], "y": spot[1]}, free_kick, header), players
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
        last_defender = is_defender(players[who]) and sum(map(is_defender, players)) == 1
        if not last_defender:
            del players[who]
    return dict(shooter), players


def consistent_shot(shot: dict, one_on_one_possible: bool = True) -> dict:
    """The shot options with the impossible combinations taken out.

    A direct free kick switches off the flags it never comes with and sets
    how the attack started. A header is almost never tagged first time in the
    data (5 headers out of 6,115), so that flag goes off too. With a
    defender in the way (see can_be_one_on_one), there's no one-on-one.
    """
    shot = dict(shot)
    if shot.get("is_free_kick"):
        shot.update(dict.fromkeys(FREE_KICK_EXCLUDES, False), play_pattern="From Free Kick")
    if shot.get("is_header"):
        shot["is_first_time"] = False
    if not one_on_one_possible:
        shot["is_one_on_one"] = False
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