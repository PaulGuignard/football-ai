-- 1. Shots and goals by type of shot
SELECT
    shot_type,
    COUNT(*) AS shots,
    COUNT(*) FILTER (WHERE shot_outcome = 'Goal') AS goals,
    ROUND(100.0 * COUNT(*) FILTER (WHERE shot_outcome = 'Goal') / COUNT(*), 1) AS conversion_pct
FROM events
WHERE type = 'Shot'
GROUP BY shot_type
ORDER BY shots DESC;

-- 2. Same by body part, without penalties (they would inflate the foot numbers)
SELECT
    shot_body_part,
    COUNT(*) AS shots,
    COUNT(*) FILTER (WHERE shot_outcome = 'Goal') AS goals,
    ROUND(100.0 * COUNT(*) FILTER (WHERE shot_outcome = 'Goal') / COUNT(*), 1) AS conversion_pct
FROM events
WHERE type = 'Shot' AND shot_type <> 'Penalty'
GROUP BY shot_body_part
ORDER BY shots DESC;

-- 3. Do the goals found in the events add up to the final scores?
WITH goals_per_match AS (
    SELECT
        match_id,
        COUNT(*) FILTER (WHERE type = 'Shot' AND shot_outcome = 'Goal') AS shot_goals,
        COUNT(*) FILTER (WHERE type = 'Own Goal For') AS own_goals
    FROM events
    GROUP BY match_id
)
SELECT
    COUNT(*) AS matches,
    COUNT(*) FILTER (WHERE m.home_score + m.away_score = g.shot_goals + g.own_goals) AS scores_ok,
    SUM(g.shot_goals) AS shot_goals,
    SUM(g.own_goals) AS own_goals
FROM matches AS m
JOIN goals_per_match AS g ON m.match_id = g.match_id;