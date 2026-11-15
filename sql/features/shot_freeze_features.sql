-- Four inputs read from the freeze frame of each shot in shot_features: how
-- many defenders block the way, how close the nearest one is, and where the
-- keeper stands. Run after shot_features.sql. Teammates are ignored.

-- Which side of the line A -> B the point P is on: positive on one side,
-- negative on the other, zero on the line (a 2D cross product)
-- (BY is an SQL keyword, hence the underscores in the names)
CREATE OR REPLACE TEMP MACRO side(p_x, p_y, a_x, a_y, b_x, b_y) AS
    (b_x - a_x) * (p_y - a_y) - (b_y - a_y) * (p_x - a_x);

-- Is P inside the triangle between the ball and the two posts? It is when it
-- sits on the same side of all three edges, whichever way round we go
CREATE OR REPLACE TEMP MACRO in_shot_cone(px, py, sx, sy) AS (
    (side(px, py, sx, sy, 120, 36) >= 0
        AND side(px, py, 120, 36, 120, 44) >= 0
        AND side(px, py, 120, 44, sx, sy) >= 0)
    OR (side(px, py, sx, sy, 120, 36) <= 0
        AND side(px, py, 120, 36, 120, 44) <= 0
        AND side(px, py, 120, 44, sx, sy) <= 0)
);

CREATE OR REPLACE TABLE shot_freeze_features AS
WITH shots AS (
    SELECT s.shot_id, e.x AS shot_x, e.y AS shot_y
    FROM shot_features AS s
    JOIN events AS e ON e.event_id = s.shot_id
),
opponents AS (
    SELECT
        f.shot_id,
        f.x,
        f.y,
        f.position = 'Goalkeeper' AS is_keeper,
        in_shot_cone(f.x, f.y, sh.shot_x, sh.shot_y) AS inside_cone,
        SQRT(POWER(f.x - sh.shot_x, 2) + POWER(f.y - sh.shot_y, 2)) AS to_shooter
    FROM shots AS sh
    JOIN shot_freeze_frames AS f ON f.shot_id = sh.shot_id
    WHERE NOT f.teammate
)
SELECT
    sh.shot_id,
    -- Outfield opponents standing between the ball and the goal
    COUNT(*) FILTER (WHERE NOT o.is_keeper AND o.inside_cone) AS defenders_in_cone,
    -- NULL when no outfield opponent is in the picture
    MIN(o.to_shooter) FILTER (WHERE NOT o.is_keeper) AS nearest_defender,
    -- Keeper inputs only when exactly one keeper is visible, NULL otherwise
    CASE WHEN COUNT(*) FILTER (WHERE o.is_keeper) = 1 THEN
        MAX(SQRT(POWER(120 - o.x, 2) + POWER(40 - o.y, 2))) FILTER (WHERE o.is_keeper)
    END AS keeper_to_goal,
    CASE WHEN COUNT(*) FILTER (WHERE o.is_keeper) = 1 THEN
        CAST(BOOL_OR(o.inside_cone) FILTER (WHERE o.is_keeper) AS INTEGER)
    END AS keeper_in_cone
FROM shots AS sh
-- LEFT JOIN: a shot with nobody in the picture still gets its row
LEFT JOIN opponents AS o ON o.shot_id = sh.shot_id
GROUP BY sh.shot_id;