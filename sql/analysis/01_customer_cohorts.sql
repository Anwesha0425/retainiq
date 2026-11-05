-- =========================================================
-- Customer Cohort Analysis
-- Grain: one row per customer per active month
-- =========================================================

DROP VIEW IF EXISTS customer_cohorts;

CREATE VIEW customer_cohorts AS

WITH valid_purchases AS (

    SELECT
        customer_id,
        invoice,
        invoice_date

    FROM stg_retail_transactions

    WHERE customer_id IS NOT NULL
      AND NOT is_cancellation
      AND quantity > 0
      AND price > 0
),

customer_first_purchase AS (

    SELECT
        customer_id,

        DATE_TRUNC(
            'month',
            MIN(invoice_date)
        )::DATE AS cohort_month

    FROM valid_purchases

    GROUP BY customer_id
),

customer_activity AS (

    SELECT DISTINCT
        vp.customer_id,

        DATE_TRUNC(
            'month',
            vp.invoice_date
        )::DATE AS activity_month

    FROM valid_purchases vp
)

SELECT
    ca.customer_id,

    cfp.cohort_month,

    ca.activity_month,

    (
        (
            EXTRACT(
                YEAR FROM ca.activity_month
            )
            -
            EXTRACT(
                YEAR FROM cfp.cohort_month
            )
        ) * 12
        +
        (
            EXTRACT(
                MONTH FROM ca.activity_month
            )
            -
            EXTRACT(
                MONTH FROM cfp.cohort_month
            )
        )
    )::INTEGER AS months_since_cohort

FROM customer_activity ca

JOIN customer_first_purchase cfp
    ON ca.customer_id = cfp.customer_id;