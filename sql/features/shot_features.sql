-- One row per shot used by the xG model, with everything the model can learn
-- from and the answer it has to predict (is_goal).
-- Penalties and shots straight from a corner are left out (see project log).
CREATE OR REPLACE TABLE shot_features AS
SELECT
    e.event_id AS shot_id,
    e.match_id,
    m.competition,
    e.team,
    e.player,

    -- What the model predicts
    CAST(e.shot_outcome = 'Goal' AS INTEGER) AS is_goal,

    -- Geometry, measured from the centre of the goal (120, 40), posts at y = 36 and 44
    SQRT(POWER(120 - e.x, 2) + POWER(40 - e.y, 2)) AS distance,
    DEGREES(ABS(ATAN2(44 - e.y, 120 - e.x) - ATAN2(36 - e.y, 120 - e.x))) AS angle,

    -- How the shot was taken
    e.shot_body_part = 'Head' AS is_header,
    e.shot_type = 'Free Kick' AS is_free_kick,
    e.shot_first_time AS is_first_time,
    e.shot_one_on_one AS is_one_on_one,
    e.under_pressure AS is_under_pressure,
    e.play_pattern,

    -- StatsBomb's own xG: never a model input, only a benchmark to compare with
    e.shot_statsbomb_xg AS statsbomb_xg
FROM events AS e
JOIN matches AS m ON e.match_id = m.match_id
WHERE e.type = 'Shot' AND e.shot_type NOT IN ('Penalty', 'Corner');