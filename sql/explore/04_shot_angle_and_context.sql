-- Same shots as in 03, with the angle and the context of each shot.
-- The angle is the width of the goal as seen from the shooter: the posts are
-- at (120, 36) and (120, 44). Wide angle = the goal looks big.
CREATE TEMP VIEW shots AS
SELECT
    shot_outcome = 'Goal' AS is_goal,
    SQRT(POWER(120 - x, 2) + POWER(40 - y, 2)) AS distance,
    DEGREES(ABS(ATAN2(44 - y, 120 - x) - ATAN2(36 - y, 120 - x))) AS angle,
    play_pattern,
    under_pressure,
    shot_first_time,
    shot_one_on_one
FROM events
WHERE type = 'Shot' AND shot_type NOT IN ('Penalty', 'Corner');

-- 1. Conversion by angle, in 10-degree bands (60 = 60 degrees and more)
SELECT
    CAST(LEAST(FLOOR(angle / 10) * 10, 60) AS INTEGER) AS angle_from,
    COUNT(*) AS shots,
    ROUND(AVG(distance), 1) AS avg_distance,
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / COUNT(*), 1) AS conversion_pct
FROM shots
GROUP BY angle_from
ORDER BY angle_from;

-- 2. Does the angle still matter at the same distance? (shots from 12 to 18 yards)
SELECT
    CAST(LEAST(FLOOR(angle / 10) * 10, 30) AS INTEGER) AS angle_from,
    COUNT(*) AS shots,
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / COUNT(*), 1) AS conversion_pct
FROM shots
WHERE distance >= 12 AND distance < 18
GROUP BY angle_from
ORDER BY angle_from;

-- 3. Where does the attack come from? (avg_distance to spot another confounder)
SELECT
    play_pattern,
    COUNT(*) AS shots,
    ROUND(AVG(distance), 1) AS avg_distance,
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / COUNT(*), 1) AS conversion_pct
FROM shots
GROUP BY play_pattern
ORDER BY shots DESC;

-- 4. Under pressure, first-time shot, one-on-one: yes or no?
SELECT 'under pressure' AS context, under_pressure AS yes, COUNT(*) AS shots,
    ROUND(AVG(distance), 1) AS avg_distance,
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / COUNT(*), 1) AS conversion_pct
FROM shots GROUP BY under_pressure
UNION ALL
SELECT 'first time', shot_first_time, COUNT(*),
    ROUND(AVG(distance), 1),
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / COUNT(*), 1)
FROM shots GROUP BY shot_first_time
UNION ALL
SELECT 'one-on-one', shot_one_on_one, COUNT(*),
    ROUND(AVG(distance), 1),
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / COUNT(*), 1)
FROM shots GROUP BY shot_one_on_one
ORDER BY context, yes;