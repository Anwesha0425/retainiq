CREATE OR REPLACE VIEW retention_dashboard AS

SELECT
    cohort_month,
    months_since_cohort,
    cohort_size,
    active_customers,
    retention_rate
FROM cohort_retention
ORDER BY
    cohort_month,
    months_since_cohort;