# football-ai

[![CI](https://github.com/PaulGuignard/football-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/PaulGuignard/football-ai/actions/workflows/ci.yml)

**Who really creates danger on a football pitch?** Goals and assists only
reward the last touch. This project gives a value to every pass, carry and
dribble according to how much it raises the chances of scoring and then
uses those values to evaluate players beyond the usual stats.

**Status.** Work in progress. The data pipeline, the exploration and the
expected goals (xG) models are done, including a version that knows where the
keeper and the defenders stand and a neural network tried against it. Every
experiment is tracked in MLflow. A demo comes next.

![Where shots are taken and how often they score](docs/figures/shot_map.png)

*The left map shows where the 37,481 shots of the four leagues were taken
from and the right one shows how often each zone scores. Goals come from a
much smaller area than shots.*

## What's done so far

- **Data.** StatsBomb open data for four complete 2015/16 seasons of the
  Premier League, La Liga, Serie A and Ligue 1. That makes 1,517 matches and
  5.3 million events loaded into a DuckDB database by an idempotent
  pipeline that can be stopped and re-run without creating duplicates.
- **Exploration in SQL.** 37,481 shots studied in `sql/explore/` with a few
  findings along the way.
  - Only 1 shot in 10 ends up in the net.
  - Headers seem to score more than foot shots overall but less at every
    distance where they are taken up to 18 yards. It is a case of Simpson's
    paradox with distance as the confounder.
  - The shot angle adds information that distance alone doesn't give.
  - Shots from counter-attacks score 15.5% of the time against about 10% for
    the others even though they are taken from about the same distance.
- **xG models.** A logistic regression and gradient boosting built first on
  eight simple inputs and then on the players' positions too, all tested on
  304 matches they never saw in training.

  | | Log loss ↓ | AUC ↑ |
  |---|---|---|
  | Naive baseline (same xG for every shot) | 0.317 | 0.500 |
  | Logistic regression | 0.264 | 0.788 |
  | Gradient boosting, default settings | 0.263 | 0.791 |
  | Gradient boosting, tuned | 0.261 | 0.796 |
  | Logistic regression with player positions | 0.257 | 0.804 |
  | **Gradient boosting with player positions** | **0.253** | **0.812** |
  | Deep Sets neural network on raw positions | 0.258 | 0.801 |
  | StatsBomb xG (professional model) | 0.250 | 0.818 |

  The logistic regression closes about 80% of the gap between the baseline
  and StatsBomb and its probabilities are well calibrated. Its coefficients
  confirm that pressure lowers the chances once distance is accounted for.
  They also hold one surprise because direct free kicks score 2 to 4 times
  more than open-play shots taken from the same distance.
  Gradient boosting tuned by cross-validation on whole matches does better
  on almost every fold but by too little to matter. The gain is about 0.001
  of log loss. With these inputs the limit is the data and not the model.
- **Players' positions.** StatsBomb's freeze frames show where everyone stands
  when the shot is taken. Four inputs computed from them in SQL count the
  defenders between the ball and the goal, measure the space around the
  shooter and say where the keeper is. They lower the log loss eight to nine
  times more than tuning did. The boosting with these inputs closes 96% of
  the gap between the baseline and StatsBomb on the test matches.

  ![Calibration with freeze frames](docs/figures/xg_calibration_v2.png)
- **A neural network on the raw positions.** A Deep Sets network in PyTorch
  reads every visible player as they are, in any order and in any number. It
  finds on its own most of what the four hand-built inputs bring and draws
  level with the logistic regression that uses them, but the boosting beats
  it on every fold. With 37,000 shots the inputs built from knowing the game
  still win, so the boosting stays the xG model of the project.
- **Experiment tracking with MLflow.** Every run keeps its settings, its
  scores and the trained model. Settings are chosen by cross-validation and
  the test set is only used once.

Every step, decision and result is written up in the
[project log](docs/project_log.md).

## Roadmap

1. ~~**Expected goals (xG) model** giving the probability that a shot
   becomes a goal with a logistic regression and then gradient boosting.~~
   Done.
2. ~~**Experiment tracking** with MLflow and tuning by cross-validation.~~
   Done.
3. ~~**xG with the players' positions** to know where the keeper and the
   defenders stand when the shot is taken, with inputs built by hand and
   with a neural network on the raw positions.~~ Done.
4. **Demo.** An app where you place a shot on a pitch and get its xG back
   from an API built with FastAPI and Docker.
5. **Action values.** The chance of scoring before and after each action.
   It is learnt from sequences of play with a small transformer.
6. **AI agent.** Ask questions about players in plain language. The agent
   queries the database, calls the models and searches football texts with
   RAG.

## Stack

Python, SQL (DuckDB), pandas, scikit-learn, PyTorch, MLflow, matplotlib /
mplsoccer, pytest, GitHub Actions, uv and Git.
Coming next are FastAPI, Docker and an LLM agent with RAG.

## Project structure

```
src/football_ai/    reusable code: features table, xG models, MLflow tracking
tests/              pytest tests, run on every pull request
scripts/            data loading, exploration, training and evaluation
sql/schema.sql      database schema
sql/features/       the tables the models learn from
sql/explore/        exploration queries
docs/               project log and figures
```

## Run it

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                     # install the environment
uv run python scripts/04_load_events.py     # load the data (~25 min)
uv run python scripts/11_load_freeze_frames.py   # load the players' positions (~20 min)
uv run python -m football_ai.features       # build the features tables
uv run python scripts/07_evaluate_xg.py     # train and compare the xG models
uv run python scripts/10_tune_boosting.py   # tune the boosting, log runs in MLflow
uv run python scripts/12_compare_feature_sets.py   # with and without positions
uv run python scripts/13_compare_deep_sets.py      # the neural network
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the runs
uv run pytest                               # run the tests
```

## Data

Data provided by [StatsBomb](https://statsbomb.com) through its
[open data](https://github.com/statsbomb/open-data) repository. The 2015/16
seasons are used to show the method and the same pipeline could run on more
recent data.