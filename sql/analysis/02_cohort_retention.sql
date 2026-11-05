-- =========================================================
-- Cohort Retention Analysis
-- Grain: one row per cohort/month combination
-- =========================================================

DROP VIEW IF EXISTS cohort_retention;

CREATE VIEW cohort_retention AS

WITH cohort_sizes AS (

    SELECT
        cohort_month,

        COUNT(DISTINCT customer_id)
            AS cohort_size

    FROM customer_cohorts

    WHERE months_since_cohort = 0

    GROUP BY cohort_month
),

monthly_activity AS (

    SELECT
        cohort_month,

        months_since_cohort,

        COUNT(DISTINCT customer_id)
            AS active_customers

    FROM customer_cohorts

    GROUP BY
        cohort_month,
        months_since_cohort
)

SELECT
    ma.cohort_month,

    ma.months_since_cohort,

    cs.cohort_size,

    ma.active_customers,

    ROUND(
        100.0
        * ma.active_customers
        / NULLIF(cs.cohort_size, 0),
        2
    ) AS retention_rate

FROM monthly_activity ma

JOIN cohort_sizes cs
    ON ma.cohort_month = cs.cohort_month

ORDER BY
    ma.cohort_month,
    ma.months_since_cohort;