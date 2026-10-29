-- Shots used for the xG model: no penalties, no shots straight from a corner.
-- Distance is measured to the centre of the goal (120, 40), in StatsBomb
-- units (roughly yards).
CREATE TEMP VIEW shots AS
SELECT
    match_id,
    shot_body_part,
    shot_type,
    shot_outcome = 'Goal' AS is_goal,
    SQRT(POWER(120 - x, 2) + POWER(40 - y, 2)) AS distance
FROM events
WHERE type = 'Shot' AND shot_type NOT IN ('Penalty', 'Corner');

-- 1. Conversion by distance, in 6-yard bands (36 = 36 yards and more)
SELECT
    CAST(LEAST(FLOOR(distance / 6) * 6, 36) AS INTEGER) AS distance_from,
    COUNT(*) AS shots,
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / COUNT(*), 1) AS conversion_pct
FROM shots
GROUP BY distance_from
ORDER BY distance_from;

-- 2. Where are headers and foot shots taken from?
SELECT
    shot_body_part,
    COUNT(*) AS shots,
    ROUND(AVG(distance), 1) AS avg_distance,
    ROUND(MEDIAN(distance), 1) AS median_distance
FROM shots
WHERE shot_body_part IN ('Head', 'Right Foot', 'Left Foot')
GROUP BY shot_body_part
ORDER BY avg_distance;

-- 3. Head vs foot at the same distance (NULLIF avoids dividing by zero)
SELECT
    CAST(LEAST(FLOOR(distance / 6) * 6, 36) AS INTEGER) AS distance_from,
    COUNT(*) FILTER (WHERE shot_body_part = 'Head') AS head_shots,
    ROUND(100.0 * COUNT(*) FILTER (WHERE shot_body_part = 'Head' AND is_goal)
        / NULLIF(COUNT(*) FILTER (WHERE shot_body_part = 'Head'), 0), 1) AS head_pct,
    COUNT(*) FILTER (WHERE shot_body_part LIKE '%Foot') AS foot_shots,
    ROUND(100.0 * COUNT(*) FILTER (WHERE shot_body_part LIKE '%Foot' AND is_goal)
        / NULLIF(COUNT(*) FILTER (WHERE shot_body_part LIKE '%Foot'), 0), 1) AS foot_pct
FROM shots
GROUP BY distance_from
ORDER BY distance_from;