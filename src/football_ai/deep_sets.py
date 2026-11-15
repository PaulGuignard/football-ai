"""xG from the raw positions of the players, with a Deep Sets neural network.

The players seen in a freeze frame form a set: their number changes from one
shot to the next and they come in no particular order. This module turns each
shot's players into fixed-size arrays with a mask saying which slots hold a
real player, then trains a network that reads every player with the same small
network and adds the results up, so that neither the order nor the number of
players matters. The eight shot inputs of v1 (distance, angle...) go in too.
"""

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from football_ai.xg import (
    CATEGORICAL_FEATURES,
    FEATURES,
    FLAG_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)

# A freeze frame never shows more than the 21 players other than the shooter
MAX_PLAYERS = 21
PLAYER_FEATURES = [
    "dx_shooter",  # where the player is, seen from the ball
    "dy_shooter",
    "dx_goal",  # and seen from the centre of the goal
    "dy_goal",
    "teammate",
    "keeper",
]
# Yards divided by 10, so that positions stay around -1 to 1 like the rest
SCALE = 10.0


def load_players(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Every player of every freeze frame, next to the location of the shot."""
    return con.sql("""
        SELECT f.shot_id, f.x, f.y, f.teammate, f.position, e.x AS shot_x, e.y AS shot_y
        FROM shot_freeze_frames AS f
        JOIN events AS e ON e.event_id = f.shot_id
    """).df()


def player_arrays(shots: pd.DataFrame, players: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The players of each shot as fixed-size arrays, in the order of `shots`.

    Returns `positions`, shaped (shots, MAX_PLAYERS, features), and `mask`,
    shaped (shots, MAX_PLAYERS), with 1 where a slot holds a real player and
    0 where it is empty padding. A shot with nobody in the picture gets an
    all-zero row and an all-zero mask.
    """
    row_of_shot = pd.Series(np.arange(len(shots)), index=shots["shot_id"].to_numpy())
    players = players[players["shot_id"].isin(row_of_shot.index)].copy()
    players["row"] = players["shot_id"].map(row_of_shot)
    # Which slot each player takes within its shot: 0, 1, 2...
    players["slot"] = players.groupby("row").cumcount()
    if (players["slot"] >= MAX_PLAYERS).any():
        raise ValueError(f"A freeze frame has more than {MAX_PLAYERS} players")

    features = np.column_stack([
        (players["x"] - players["shot_x"]) / SCALE,
        (players["y"] - players["shot_y"]) / SCALE,
        (120 - players["x"]) / SCALE,
        (40 - players["y"]) / SCALE,
        players["teammate"].astype(float),
        players["position"].eq("Goalkeeper").astype(float),
    ]).astype(np.float32)

    positions = np.zeros((len(shots), MAX_PLAYERS, len(PLAYER_FEATURES)), dtype=np.float32)
    mask = np.zeros((len(shots), MAX_PLAYERS), dtype=np.float32)
    rows, slots = players["row"].to_numpy(), players["slot"].to_numpy()
    positions[rows, slots] = features
    mask[rows, slots] = 1.0
    return positions, mask


class DeepSetsNetwork(nn.Module):
    """Same small network for every player, a sum over players, then the shot.

    `encode_player` turns each player into `hidden` numbers. Summing them over
    the real players gives a summary of the picture that doesn't depend on
    their order or on how many there are. `predict` reads that summary next
    to the shot's own inputs and gives one number per shot, the log-odds of a
    goal (the sigmoid of it is the xG).
    """

    def __init__(self, n_shot_inputs: int, hidden: int = 32):
        super().__init__()
        self.encode_player = nn.Sequential(
            nn.Linear(len(PLAYER_FEATURES), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.predict = nn.Sequential(
            nn.Linear(hidden + n_shot_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, positions, mask, shot_inputs):
        players = self.encode_player(positions)  # (shots, slots, hidden)
        # Empty slots are multiplied by 0, so they add nothing to the sum
        picture = (players * mask.unsqueeze(-1)).sum(dim=1)
        return self.predict(torch.cat([picture, shot_inputs], dim=1)).squeeze(1)


@dataclass
class DeepSetsXG:
    """A trained network with the preparation of the shot inputs it was trained with."""

    preparation: ColumnTransformer
    network: DeepSetsNetwork
    best_epoch: int

    def predict_xg(self, shots: pd.DataFrame, positions: np.ndarray, mask: np.ndarray) -> pd.Series:
        """The xG of each shot. positions and mask must be row-aligned with shots."""
        self.network.eval()
        with torch.no_grad():
            logits = self.network(*_tensors(self.preparation, shots, positions, mask))
        return pd.Series(torch.sigmoid(logits).numpy(), index=shots.index, name="xg")


def _shot_preparation() -> ColumnTransformer:
    """The v1 shot inputs prepared as for the logistic regression, as plain numbers."""
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("flags", "passthrough", FLAG_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])


def _tensors(preparation, shots, positions, mask):
    """What the network reads, as float32 tensors."""
    shot_inputs = preparation.transform(shots[FEATURES]).astype(np.float32)
    return torch.from_numpy(positions), torch.from_numpy(mask), torch.from_numpy(shot_inputs)


def train_deep_sets(
    shots: pd.DataFrame,
    positions: np.ndarray,
    mask: np.ndarray,
    seed: int = 42,
    hidden: int = 32,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    patience: int = 10,
    max_epochs: int = 200,
) -> DeepSetsXG:
    """Train the network on these shots, with early stopping on 10% of their matches.

    positions and mask must be row-aligned with shots. Every epoch goes once
    through the training shots in random batches, then scores the held-back
    matches. Training stops after `patience` epochs without progress there and
    the best weights seen are kept. The seed fixes the held-back matches, the
    starting weights and the batches, so the same call gives the same model.
    """
    torch.manual_seed(seed)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    learn, check = next(splitter.split(shots, groups=shots["match_id"]))

    preparation = _shot_preparation().fit(shots.iloc[learn][FEATURES])
    learn_x = _tensors(preparation, shots.iloc[learn], positions[learn], mask[learn])
    check_x = _tensors(preparation, shots.iloc[check], positions[check], mask[check])
    learn_y = torch.tensor(shots.iloc[learn][TARGET].to_numpy(), dtype=torch.float32)
    check_y = torch.tensor(shots.iloc[check][TARGET].to_numpy(), dtype=torch.float32)

    network = DeepSetsNetwork(n_shot_inputs=learn_x[2].shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    # Log loss, fed with log-odds rather than probabilities: more stable
    loss_fn = nn.BCEWithLogitsLoss()

    best_loss, best_weights, best_epoch = float("inf"), None, 0
    for epoch in range(max_epochs):
        network.train()
        for batch in torch.randperm(len(learn_y)).split(batch_size):
            optimizer.zero_grad()
            loss = loss_fn(network(*(x[batch] for x in learn_x)), learn_y[batch])
            loss.backward()  # backpropagation: how each weight should move
            optimizer.step()

        network.eval()
        with torch.no_grad():
            check_loss = loss_fn(network(*check_x), check_y).item()
        if check_loss < best_loss:
            best_loss, best_epoch = check_loss, epoch
            best_weights = {k: v.clone() for k, v in network.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    network.load_state_dict(best_weights)
    return DeepSetsXG(preparation, network, best_epoch)


def cross_validate_deep_sets(
    shots: pd.DataFrame, positions: np.ndarray, mask: np.ndarray, n_splits: int = 5, **settings
) -> np.ndarray:
    """Log loss of the network on each of n_splits folds of whole matches.

    Same folds as football_ai.xg.cross_validate_by_match on the same shots,
    so the network can be compared with the other models fold by fold.
    """
    losses = []
    folds = GroupKFold(n_splits=n_splits).split(shots, groups=shots["match_id"])
    for learn_idx, check_idx in folds:
        model = train_deep_sets(
            shots.iloc[learn_idx], positions[learn_idx], mask[learn_idx], **settings
        )
        check = shots.iloc[check_idx]
        xg = model.predict_xg(check, positions[check_idx], mask[check_idx])
        losses.append(log_loss(check[TARGET], xg, labels=[0, 1]))
    return np.array(losses)