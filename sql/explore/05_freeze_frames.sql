-- What the freeze frames add: defenders in the way and where the keeper is.
-- Same 37,481 shots as the model, with the inputs from shot_freeze_features.
-- The "cone" is the triangle between the ball and the two posts.
CREATE TEMP VIEW shots AS
SELECT s.is_goal, s.distance, f.*
FROM shot_features AS s
JOIN shot_freeze_features AS f ON f.shot_id = s.shot_id;

-- 1. Conversion by number of defenders in the cone (3 = three or more)
SELECT
    LEAST(defenders_in_cone, 3) AS defenders_in_cone,
    COUNT(*) AS shots,
    ROUND(AVG(distance), 1) AS avg_distance,
    ROUND(100.0 * AVG(is_goal), 1) AS conversion_pct
FROM shots
GROUP BY ALL
ORDER BY 1;

-- 2. Same question at the same distance (shots from 12 to 18 yards)
SELECT
    LEAST(defenders_in_cone, 3) AS defenders_in_cone,
    COUNT(*) AS shots,
    ROUND(100.0 * AVG(is_goal), 1) AS conversion_pct
FROM shots
WHERE distance BETWEEN 12 AND 18
GROUP BY ALL
ORDER BY 1;

-- 3. Is the keeper between the ball and the goal? (NULL = no single keeper visible)
SELECT
    keeper_in_cone,
    COUNT(*) AS shots,
    ROUND(AVG(distance), 1) AS avg_distance,
    ROUND(100.0 * AVG(is_goal), 1) AS conversion_pct
FROM shots
GROUP BY ALL
ORDER BY 1;

-- 4. How far the keeper stands from the centre of his goal, in 2-yard bands
SELECT
    CAST(LEAST(FLOOR(keeper_to_goal / 2) * 2, 8) AS INTEGER) AS keeper_yards_from,
    COUNT(*) AS shots,
    ROUND(AVG(distance), 1) AS avg_distance,
    ROUND(100.0 * AVG(is_goal), 1) AS conversion_pct
FROM shots
WHERE keeper_to_goal IS NOT NULL
GROUP BY ALL
ORDER BY 1;

-- 5. Space around the shooter: nearest outfield opponent, 12 to 18 yards only
SELECT
    CAST(LEAST(FLOOR(nearest_defender), 5) AS INTEGER) AS nearest_yards_from,
    COUNT(*) AS shots,
    ROUND(100.0 * AVG(is_goal), 1) AS conversion_pct
FROM shots
WHERE distance BETWEEN 12 AND 18 AND nearest_defender IS NOT NULL
GROUP BY ALL
ORDER BY 1;