-- =========================================================
-- Customer RFM Analysis
-- Grain: one row per customer
-- =========================================================

DROP VIEW IF EXISTS customer_rfm;

CREATE VIEW customer_rfm AS

WITH valid_purchases AS (

    SELECT
        customer_id,
        invoice,
        invoice_date,
        line_amount

    FROM stg_retail_transactions

    WHERE customer_id IS NOT NULL

      -- Exclude cancellations
      AND NOT is_cancellation

      -- Exclude returns
      AND quantity > 0

      -- Exclude zero/negative price records
      AND price > 0
),

reference_date AS (

    SELECT
        MAX(invoice_date)::DATE AS analysis_date

    FROM valid_purchases
),

customer_metrics AS (

    SELECT
        customer_id,

        -- Most recent purchase date
        MAX(invoice_date)::DATE
            AS last_purchase_date,

        -- Number of distinct purchase invoices
        COUNT(DISTINCT invoice)
            AS frequency,

        -- Total positive purchase value
        SUM(line_amount)
            AS monetary

    FROM valid_purchases

    GROUP BY customer_id
)

SELECT
    cm.customer_id,

    cm.last_purchase_date,

    rd.analysis_date,

    -- Days since last purchase
    (
        rd.analysis_date
        - cm.last_purchase_date
    ) AS recency,

    cm.frequency,

    cm.monetary

FROM customer_metrics cm

CROSS JOIN reference_date rd;