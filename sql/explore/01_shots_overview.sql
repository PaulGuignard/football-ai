-- 1. How much data did we load?
SELECT
    COUNT(DISTINCT match_id) AS matches,
    COUNT(*) AS events
FROM events;

-- 2. The most frequent event types
SELECT type, COUNT(*) AS n
FROM events
GROUP BY type
ORDER BY n DESC
LIMIT 10;

-- 3. How many shots, and how many goals?
SELECT
    COUNT(*) AS shots,
    COUNT(*) FILTER (WHERE shot_outcome = 'Goal') AS goals
FROM events
WHERE type = 'Shot';

-- 4. What happens to a shot?
SELECT
    shot_outcome,
    COUNT(*) AS shots,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM events WHERE type = 'Shot'), 1) AS pct
FROM events
WHERE type = 'Shot'
GROUP BY shot_outcome
ORDER BY shots DESC;