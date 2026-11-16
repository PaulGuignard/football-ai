"""A small web API that gives the xG of a shot.

Send a shot as JSON (where it is taken from, how, and where the other players
stand) to POST /xg and get its xG back, with the inputs the model saw. The
model is the file written by scripts/15_export_model.py, loaded once when the
API starts. Interactive docs are served at /docs.

Usage:
    uv run uvicorn football_ai.api:app
"""

import json
import math
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from football_ai.model_store import MODEL_PATH, load_model
from football_ai.shot_inputs import shot_row
from football_ai.xg import predict_xg

# The ways an attack can start, as StatsBomb names them
PlayPattern = Literal[
    "Regular Play", "From Free Kick", "From Throw In", "From Corner", "From Counter",
    "From Goal Kick", "From Keeper", "From Kick Off", "Other",
]

model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the API starts, not on every request."""
    model["xg"] = load_model(MODEL_PATH)
    model["details"] = json.loads(MODEL_PATH.with_suffix(".json").read_text())
    yield
    model.clear()


app = FastAPI(
    title="football-ai xG",
    description="The chance that a shot ends up in the net, from where it's taken and who's around.",
    lifespan=lifespan,
)


class Player(BaseModel):
    """Someone in the picture other than the shooter. The pitch is 120 x 80."""

    x: float = Field(ge=0, le=120)
    y: float = Field(ge=0, le=80)
    teammate: bool = False
    keeper: bool = False


class Shot(BaseModel):
    """A shot, with the shooting team attacking towards x = 120."""

    x: float = Field(ge=0, le=120, examples=[108.0])
    y: float = Field(ge=0, le=80, examples=[40.0])
    is_header: bool = False
    is_free_kick: bool = False
    is_first_time: bool = False
    is_one_on_one: bool = False
    is_under_pressure: bool = False
    play_pattern: PlayPattern = "Regular Play"
    # A freeze frame never shows more than the 21 other players
    players: list[Player] = Field(default_factory=list, max_length=21)


class Prediction(BaseModel):
    """The xG, and the inputs it was computed from (None where unknown)."""

    xg: float
    inputs: dict[str, float | bool | str | None]


@app.get("/health")
def health() -> dict:
    """Is the API up, and which model does it serve?"""
    return {"status": "ok", "model": model["details"]}


@app.post("/xg")
def xg(shot: Shot) -> Prediction:
    """The xG of one shot."""
    row = shot_row(
        shot.x,
        shot.y,
        [player.model_dump() for player in shot.players],
        is_header=shot.is_header,
        is_free_kick=shot.is_free_kick,
        is_first_time=shot.is_first_time,
        is_one_on_one=shot.is_one_on_one,
        is_under_pressure=shot.is_under_pressure,
        play_pattern=shot.play_pattern,
    )
    inputs = row.iloc[0].to_dict()
    # JSON has no NaN: a missing input goes out as null
    inputs = {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in inputs.items()}
    return Prediction(xg=float(predict_xg(model["xg"], row).iloc[0]), inputs=inputs)