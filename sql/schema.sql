-- Only the columns we need for xG and action values (39 out of ~90).
-- Skipped for now: shot freeze frames, tactics, goalkeeper details.

CREATE TABLE IF NOT EXISTS matches (
    match_id     INTEGER PRIMARY KEY,
    competition  VARCHAR,
    season       VARCHAR,
    match_date   DATE,
    home_team    VARCHAR,
    away_team    VARCHAR,
    home_score   INTEGER,
    away_score   INTEGER
);

CREATE TABLE IF NOT EXISTS events (
    -- Context: when, who and what
    match_id          INTEGER,
    event_id          VARCHAR,
    event_index       INTEGER,   -- order of the event within the match
    period            INTEGER,
    minute            INTEGER,
    second            INTEGER,
    possession        INTEGER,   -- possession number within the match
    possession_team   VARCHAR,
    play_pattern      VARCHAR,
    team              VARCHAR,
    player            VARCHAR,
    position          VARCHAR,
    type              VARCHAR,
    x                 DOUBLE,    -- pitch is 120 x 80, team attacks towards x = 120
    y                 DOUBLE,
    duration          DOUBLE,
    under_pressure    BOOLEAN,

    -- Passes
    pass_end_x        DOUBLE,
    pass_end_y        DOUBLE,
    pass_length       DOUBLE,
    pass_height       VARCHAR,
    pass_body_part    VARCHAR,
    pass_type         VARCHAR,
    pass_outcome      VARCHAR,   -- NULL means the pass was completed
    pass_shot_assist  BOOLEAN,
    pass_goal_assist  BOOLEAN,

    -- Carries and dribbles
    carry_end_x       DOUBLE,
    carry_end_y       DOUBLE,
    dribble_outcome   VARCHAR,

    -- Shots
    shot_end_x        DOUBLE,
    shot_end_y        DOUBLE,
    shot_outcome      VARCHAR,
    shot_type         VARCHAR,
    shot_body_part    VARCHAR,
    shot_technique    VARCHAR,
    shot_first_time   BOOLEAN,
    shot_one_on_one   BOOLEAN,
    shot_statsbomb_xg DOUBLE,    -- StatsBomb's own xG, used as a benchmark
    shot_key_pass_id  VARCHAR    -- event_id of the pass that led to the shot
);