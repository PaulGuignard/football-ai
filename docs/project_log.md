# Project log

Notes on how the project is going: the questions I ask myself, what I tried,
what I found and the decisions that came out of it. Newest entries at the bottom.

## Why this project

Goals and assists only reward the last touch. The pass that breaks a line
three actions before a goal shows up nowhere in the usual stats. The question
behind this project: **who really creates danger on a football pitch?**

The idea is to give every on-ball action (pass, carry, dribble) a value: how
much it changes the chances of scoring in the next few moments. Then to build
an assistant that can answer questions about players using these values.

The plan, step by step:

1. **An expected goals (xG) model**: the probability that a shot ends up in
   the net. It's the simplest version of the main question (how likely is a
   goal from this situation?), and it lets me build the whole chain (data,
   training, evaluation) on a smaller problem first.
2. **Action values**: estimate the chances of scoring before and after each
   action, the difference being the value of the action. A sequence model
   (a small transformer) will read the last actions to judge the current one.
3. **An AI agent**: a language model (LLM) that decides by itself which tool
   to use to answer a question. Its tools: SQL queries on the database, the
   models above (served through an API), and a RAG search over football texts
   (event definitions, match reports...) so it can explain and quote its
   sources instead of making things up.

It's also my way of practising deep learning, generative AI (RAG, agents),
SQL and MLOps on a real problem rather than on toy examples.

---

## Choosing the data (27/10)

StatsBomb open data lists 80 competition-seasons, but a listed season is not
necessarily a complete one. Counting matches and teams per season
(`scripts/02_count_matchs.py`) showed two kinds of seasons:

| Season | Matches | Teams | |
|---|---|---|---|
| Premier League, La Liga, Serie A 2015/16 | 380 | 20 | complete |
| Ligue 1 2015/16 | 377 | 20 | 3 matches missing |
| Bundesliga 2015/16 | 34 | 18 | partial |
| La Liga 2020/21, Ligue 1 2022/23, Bundesliga 2023/24 | 32 to 35 | 18 to 20 | partial |

The team count is misleading here: 34 matches with 18 teams is exactly one
team's season (17 opponents, home and away). These partial seasons only
follow one club, usually a strong one, so a model trained on them would learn
how that club plays rather than how a league plays.

**Decision:** train on the four complete 2015/16 seasons (1,517 matches).
World Cups and Euros are complete too, but I keep them aside for now.

## How the model will be evaluated (27/10, plan)

- Main test set: club matches from 2015/16 that the model never sees during
  training, split **by match** rather than by shot, so that shots from the
  same game can't end up on both sides.
- Possible second test on World Cups / Euros, to see if the model still works
  on national teams. Its limit: it changes two things at once (clubs vs
  national teams, 2015/16 vs 2018-2024), so a drop in performance couldn't be
  attributed to one of them.

## A first look at a match (27/10)

One match (Chelsea 2-0 Arsenal, September 2015) is 3,732 events and 90
columns. A few things worth knowing before going further:

- Passes, ball receipts and carries make up about 3/4 of the events. There
  were only 33 shots.
- Some events are not actions at all (`Starting XI`, `Half Start`...): no
  player, no location. They will have to be filtered out.
- One of the two goals was an own goal, which is not a shot. Goals in a match
  are not always shots.
- Coordinates are given from the point of view of the team making the action
  (always attacking towards x = 120). Comparing both teams in the same
  sequence means flipping one of them: (x, y) becomes (120 - x, 80 - y).

## Loading everything into DuckDB (28/10)

About 5 million events is too much to keep comfortably in pandas on a laptop,
so the four seasons go into a DuckDB database (a single file, no server).
`scripts/04_load_events.py` downloads each match, cleans it and inserts it.

- Only 39 of the ~90 columns are kept (see `sql/schema.sql`). What I kept:
  the context needed to rebuild sequences of play (match, period, time,
  possession, team, player, location) and the details of the actions the
  project is about: passes, carries, dribbles and shots. What I left out:
  details of duels, fouls, cards, substitutions and goalkeeper actions, which
  don't say how dangerous an action is, plus tactics and shot freeze frames,
  which are nested structures. Freeze frames (where every player stands when a
  shot is taken) could improve the xG model later; adding a column only means
  re-running the loading script.
- Missing text has to become `NULL`, not the string `"nan"`.
- StatsBomb only fills flags like `under_pressure` when they are true, so an
  empty cell means `False`.
- Matches already in the database are skipped, so the script can be stopped
  and run again without duplicates. Each match is inserted in one go, so a
  match is never half loaded.

Full run: about 25 minutes, 1,517 matches, 5,321,459 events.

## First SQL look at shots (29/10)

From now on, queries live in `.sql` files under `sql/explore/` so they can be
read directly on GitHub. `scripts/run_sql.py` runs a file and prints each
result as a table, with the comment above the query as its title.

`sql/explore/01_shots_overview.sql`

- 37,888 shots, about 25 per match.
- 3,869 goals: **10.2%** of shots end up in the net.
- Other outcomes: off target 33%, blocked 25%, saved 23%, wayward 6%, post 2%.

What it means for the xG model (step 1 of the plan):

- The target is goal / no goal, and the classes are imbalanced (10% goals).
  A model always answering "no goal" would be right 90% of the time and
  still useless, so accuracy is not the right metric. I'll use log loss, the
  Brier score and calibration instead.
- First baseline: give every shot an xG of 0.10. Any real model has to do
  clearly better than that.

Open questions for the next step:

- Penalties are counted as shots and they go in far more often. Should they
  be part of the model?
- Do the match scores match the goals from shots plus own goals?

## Shot types, body parts and a data check (29/10)

`sql/explore/02_shot_types_and_scores.sql`

| Shot type | Shots | Goals | Conversion |
|---|---|---|---|
| Open play | 35,732 | 3,453 | 9.7% |
| Direct free kick | 1,749 | 113 | 6.5% |
| Penalty | 400 | 300 | 75.0% |
| Direct from a corner | 7 | 3 | 42.9% |

- Penalties go in 3 times out of 4. It's always the same situation (11 m,
  facing the keeper), so there's nothing for a model to learn from their
  position, and they would blur what it learns from other shots. **Decision:**
  penalties are left out of training and get a fixed xG of 0.75.
- "Corner" here means a shot taken straight from the corner flag, not a shot
  after a corner (those are open play, with `play_pattern = 'From Corner'`).
  Only 7 of them: a model can't learn anything from 7 examples, so I leave
  them out too. Free kicks stay: 1,749 shots is enough, and the shot type
  will be one of the model's inputs.
- That leaves 37,481 shots to train the xG model.

Body parts, penalties excluded: right foot 9.4%, left foot 8.9%, **head
10.7%**. I expected feet to do better. My guess: headers are almost always
taken close to goal, while foot shots include lots of long-range attempts.
If that's right, distance is a confounder (it drives both the body part used
and the chance of scoring), and at equal distance headers should actually do
worse. Next step: check it.

Data check: for all 1,517 matches, goals from shots (3,869) plus own goals
(128) add up exactly to the final scores.

## Distance, and why headers looked better than they are (29/10)

`sql/explore/03_shot_distance.sql`

Distance is measured from the shot to the centre of the goal, in StatsBomb
units (roughly yards). Penalties and shots straight from a corner are left
out, which leaves the 37,481 shots of the model (the seven distance bands
add up to exactly that).

| Distance (yards) | Shots | Conversion |
|---|---|---|
| 0-6 | 1,022 | 44.7% |
| 6-12 | 7,624 | 17.7% |
| 12-18 | 8,398 | 12.0% |
| 18-24 | 8,220 | 5.5% |
| 24-30 | 7,721 | 2.9% |
| 30-36 | 3,569 | 1.6% |
| 36+ | 927 | 1.7% |

The chance of scoring drops very fast in the first yards, then flattens out
around 2%. The relationship is clearly not a straight line, which matters
for the model: a logistic regression on raw distance may struggle with it.

Headers are taken from about 10 yards on average, foot shots from about
21. And at the same distance, feet win every time:

| Distance (yards) | Headers | Head conversion | Foot shots | Foot conversion |
|---|---|---|---|---|
| 0-6 | 498 | 32.7% | 494 | 56.1% |
| 6-12 | 4,169 | 10.6% | 3,410 | 26.5% |
| 12-18 | 1,392 | 3.5% | 6,997 | 13.7% |

So the guess was right. Overall, headers score more often (10.7% vs about
9%), but within every distance band they score 2 to 4 times less. Headers
only looked better because they are taken much closer to goal. This is a
textbook case of Simpson's paradox: a trend that holds in each group flips
when the groups are pooled.

**Takeaway for the model:** distance is the most important input, and body
part has to be used together with it, never on its own. Looking at one
variable at a time would have given the wrong conclusion.