-- =========================================================
-- Inactivity-Based Customer Churn
-- Grain: one row per customer
-- =========================================================

DROP VIEW IF EXISTS customer_churn;

CREATE VIEW customer_churn AS

WITH valid_purchases AS (

    SELECT
        customer_id,
        invoice_date,
        line_amount

    FROM stg_retail_transactions

    WHERE customer_id IS NOT NULL
      AND NOT is_cancellation
      AND quantity > 0
      AND price > 0
),

dataset_end AS (

    SELECT
        MAX(invoice_date)::DATE AS observation_end_date

    FROM valid_purchases
),

customer_activity AS (

    SELECT
        customer_id,

        MIN(invoice_date)::DATE
            AS first_purchase_date,

        MAX(invoice_date)::DATE
            AS last_purchase_date,

        COUNT(DISTINCT invoice)
            AS purchase_frequency,

        SUM(line_amount)
            AS total_monetary_value

    FROM (
        SELECT
            customer_id,
            invoice_date,
            line_amount,
            invoice
        FROM stg_retail_transactions

        WHERE customer_id IS NOT NULL
          AND NOT is_cancellation
          AND quantity > 0
          AND price > 0
    ) purchases

    GROUP BY customer_id
)

SELECT
    ca.customer_id,

    ca.first_purchase_date,
    ca.last_purchase_date,

    de.observation_end_date,

    (
        de.observation_end_date
        - ca.last_purchase_date
    ) AS days_since_last_purchase,

    ca.purchase_frequency,
    ca.total_monetary_value,

    CASE
        WHEN (
            de.observation_end_date
            - ca.last_purchase_date
        ) >= 90
        THEN 1
        ELSE 0
    END AS churn

FROM customer_activity ca

CROSS JOIN dataset_end de;