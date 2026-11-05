-- =========================================================
-- Time-Based Churn Modeling Dataset
--
-- Observation period:
--     Up to 2011-09-01
--
-- Prediction horizon:
--     90 days after cutoff
--
-- Grain:
--     one row per customer
--
-- Important:
--     Features use ONLY transactions before the cutoff.
--     Churn is determined ONLY from transactions after
--     the cutoff.
-- =========================================================


DROP VIEW IF EXISTS time_based_churn_model;


CREATE VIEW time_based_churn_model AS

WITH parameters AS (

    SELECT
        DATE '2011-09-01' AS cutoff_date,

        DATE '2011-09-01'
            + INTERVAL '90 days'
            AS prediction_end_date
),


-- =========================================================
-- Historical purchases
-- =========================================================

historical_purchases AS (

    SELECT
        r.customer_id,
        r.invoice,
        r.invoice_date,
        r.line_amount,
        r.quantity,
        r.stock_code

    FROM stg_retail_transactions r

    CROSS JOIN parameters p

    WHERE r.customer_id IS NOT NULL

      AND NOT r.is_cancellation

      AND r.quantity > 0

      AND r.price > 0

      AND r.invoice_date < p.cutoff_date
),


-- =========================================================
-- Future purchases used ONLY for target creation
-- =========================================================

future_purchases AS (

    SELECT DISTINCT
        r.customer_id

    FROM stg_retail_transactions r

    CROSS JOIN parameters p

    WHERE r.customer_id IS NOT NULL

      AND NOT r.is_cancellation

      AND r.quantity > 0

      AND r.price > 0

      AND r.invoice_date >= p.cutoff_date

      AND r.invoice_date < p.prediction_end_date
),


-- =========================================================
-- Customer behavioral features
-- =========================================================

customer_features AS (

    SELECT
        hp.customer_id,

        COUNT(DISTINCT hp.invoice)
            AS purchase_frequency,

        SUM(hp.line_amount)
            AS total_spend,

        AVG(hp.line_amount)
            AS average_line_value,

        COUNT(
            DISTINCT DATE_TRUNC(
                'month',
                hp.invoice_date
            )
        ) AS active_purchase_months,

        COUNT(DISTINCT hp.stock_code)
            AS unique_products_purchased,

        SUM(hp.quantity)
            AS total_units_purchased,

        COUNT(*) AS total_purchase_lines,

        MIN(hp.invoice_date)::DATE
            AS first_purchase_date,

        MAX(hp.invoice_date)::DATE
            AS last_purchase_date

    FROM historical_purchases hp

    GROUP BY hp.customer_id
),


-- =========================================================
-- Historical return behavior
-- =========================================================

customer_returns AS (

    SELECT
        r.customer_id,

        COUNT(*) AS return_lines,

        ABS(
            SUM(
                CASE
                    WHEN r.quantity < 0
                    THEN r.quantity
                    ELSE 0
                END
            )
        ) AS returned_units

    FROM stg_retail_transactions r

    CROSS JOIN parameters p

    WHERE r.customer_id IS NOT NULL

      AND r.quantity < 0

      AND r.invoice_date < p.cutoff_date

    GROUP BY r.customer_id
)


-- =========================================================
-- Final modeling dataset
-- =========================================================

SELECT
    cf.customer_id,

    -- Behavioral features
    cf.purchase_frequency,

    ROUND(
        cf.total_spend,
        2
    ) AS total_spend,

    ROUND(
        cf.average_line_value,
        2
    ) AS average_line_value,

    cf.active_purchase_months,

    cf.unique_products_purchased,

    cf.total_units_purchased,

    cf.total_purchase_lines,

    COALESCE(
        cr.return_lines,
        0
    ) AS return_lines,

    COALESCE(
        cr.returned_units,
        0
    ) AS returned_units,

    ROUND(
        cf.total_spend
        / NULLIF(
            cf.purchase_frequency,
            0
        ),
        2
    ) AS average_order_value,

    ROUND(
        1.0
        * COALESCE(
            cr.return_lines,
            0
        )
        / NULLIF(
            cf.total_purchase_lines,
            0
        ),
        4
    ) AS return_line_rate,

    -- Useful metadata
    cf.first_purchase_date,

    cf.last_purchase_date,

    p.cutoff_date,

    p.prediction_end_date,

    -- =====================================================
    -- Target
    --
    -- 0 = customer purchased during the next 90 days
    -- 1 = customer did NOT purchase during the next 90 days
    -- =====================================================

    CASE
        WHEN fp.customer_id IS NULL
        THEN 1
        ELSE 0
    END AS churn

FROM customer_features cf

LEFT JOIN customer_returns cr
    ON cf.customer_id = cr.customer_id

LEFT JOIN future_purchases fp
    ON cf.customer_id = fp.customer_id

CROSS JOIN parameters p;