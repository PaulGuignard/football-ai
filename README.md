# football-ai

[![CI](https://github.com/PaulGuignard/football-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/PaulGuignard/football-ai/actions/workflows/ci.yml)

**Who really creates danger on a football pitch?** Goals and assists only
reward the last touch. This project gives a value to every on-ball action
(pass, carry, dribble) according to how much it raises the chances of
scoring, then uses those values to evaluate players beyond the usual stats.

**Status:** work in progress. The data pipeline, the exploration and a first
expected goals (xG) model are done; more advanced models are next.

![Where shots are taken and how often they score](docs/figures/shot_map.png)

*Left: where the 37,481 shots of the four leagues were taken from. Right:
how often each zone scores. Goals come from a much smaller area than shots.*

## What's done so far

- **Data**: StatsBomb open data, four complete 2015/16 seasons (Premier
  League, La Liga, Serie A, Ligue 1): 1,517 matches, 5.3 million events,
  loaded into a DuckDB database by an idempotent pipeline (it can be stopped
  and re-run without duplicates).
- **Exploration in SQL** (`sql/explore/`): 37,481 shots studied. A few
  findings:
  - only 1 shot in 10 ends up in the net;
  - headers seem to score more than foot shots overall, but less at every
    distance band where they are taken (up to 18 yards): a case of
    Simpson's paradox, distance being the confounder;
  - the shot angle adds information that distance alone doesn't;
  - shots from counter-attacks score more often (15.5% vs about 10%), from
    about the same average distance.
- **First xG model**: a logistic regression on eight simple inputs, tested on
  304 matches it never saw during training.

  | | Log loss ↓ | AUC ↑ |
  |---|---|---|
  | Naive baseline (same xG for every shot) | 0.317 | 0.500 |
  | **Logistic regression** | **0.264** | **0.788** |
  | StatsBomb xG (professional model) | 0.250 | 0.818 |

  It closes about 80% of the gap between the baseline and StatsBomb, with
  well-calibrated probabilities. Its coefficients confirm what the
  exploration suggested (pressure lowers the chances once distance is
  accounted for) and show one surprise: at the same distance, direct free
  kicks score 2 to 4 times more than open-play shots.

Every step, decision and result is written up in the
[project log](docs/project_log.md).

## Roadmap

1. ~~**Expected goals (xG) model**, the probability that a shot becomes a
   goal: logistic regression as a baseline.~~ Done.
2. **Better xG models**: gradient boosting, then a small neural network in
   PyTorch, with experiment tracking (MLflow); then the keeper's and
   defenders' positions at the moment of the shot.
3. **Demo**: an app where you place a shot on a pitch and get its xG, served
   by an API (FastAPI, Docker).
4. **Action values**: the chance of scoring before and after each action,
   learnt from sequences of play with a small transformer.
5. **AI agent**: ask questions about players in plain language; the agent
   queries the database, calls the models and searches football texts (RAG).

## Stack

Python, SQL (DuckDB), pandas, scikit-learn, matplotlib / mplsoccer, pytest,
GitHub Actions, uv, Git.
Coming next: PyTorch, MLflow, FastAPI, Docker, an LLM agent with RAG.

## Project structure

```
src/football_ai/    reusable code: features table, xG model
tests/              pytest tests, run on every pull request
scripts/            data loading, exploration, training and evaluation
sql/schema.sql      database schema
sql/features/       the table the models learn from
sql/explore/        exploration queries
docs/               project log and figures
```

## Run it

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                     # install the environment
uv run python scripts/04_load_events.py     # load the data (~25 min)
uv run python -m football_ai.features       # build the features table
uv run python scripts/07_evaluate_xg.py     # train and evaluate the xG model
uv run pytest                               # run the tests
```

## Data

Data provided by [StatsBomb](https://statsbomb.com) through its
[open data](https://github.com/statsbomb/open-data) repository. The 2015/16
seasons are used to demonstrate the method; the same pipeline could run on
recent data.