-- =========================================================
-- Customer Behavioral Features
-- Grain: one row per customer
-- Purpose: features for analysis and churn modeling
-- =========================================================

DROP VIEW IF EXISTS customer_behavior_features;

CREATE VIEW customer_behavior_features AS

WITH valid_purchases AS (

    SELECT
        customer_id,
        invoice,
        invoice_date,
        line_amount,
        quantity,
        stock_code
    FROM stg_retail_transactions

    WHERE customer_id IS NOT NULL
      AND NOT is_cancellation
      AND quantity > 0
      AND price > 0
),

customer_purchase_features AS (

    SELECT
        customer_id,

        COUNT(DISTINCT invoice)
            AS purchase_frequency,

        SUM(line_amount)
            AS total_spend,

        AVG(line_amount)
            AS average_line_value,

        COUNT(DISTINCT DATE_TRUNC(
            'month',
            invoice_date
        ))
            AS active_purchase_months,

        COUNT(DISTINCT stock_code)
            AS unique_products_purchased,

        MIN(invoice_date)::DATE
            AS first_purchase_date,

        MAX(invoice_date)::DATE
            AS last_purchase_date,

        SUM(quantity)
            AS total_units_purchased,

        COUNT(*) AS total_purchase_lines

    FROM valid_purchases

    GROUP BY customer_id
),

customer_return_features AS (

    SELECT
        customer_id,

        COUNT(*) AS return_lines,

        ABS(
            SUM(
                CASE
                    WHEN quantity < 0
                    THEN quantity
                    ELSE 0
                END
            )
        ) AS returned_units

    FROM stg_retail_transactions

    WHERE customer_id IS NOT NULL
      AND quantity < 0

    GROUP BY customer_id
)

SELECT
    cpf.customer_id,

    cpf.purchase_frequency,

    ROUND(
        cpf.total_spend,
        2
    ) AS total_spend,

    ROUND(
        cpf.average_line_value,
        2
    ) AS average_line_value,

    cpf.active_purchase_months,

    cpf.unique_products_purchased,

    cpf.total_units_purchased,

    cpf.total_purchase_lines,

    cpf.first_purchase_date,

    cpf.last_purchase_date,

    COALESCE(
        crf.return_lines,
        0
    ) AS return_lines,

    COALESCE(
        crf.returned_units,
        0
    ) AS returned_units,

    ROUND(
        cpf.total_spend
        / NULLIF(
            cpf.purchase_frequency,
            0
        ),
        2
    ) AS average_order_value,

    ROUND(
        1.0 * COALESCE(
            crf.return_lines,
            0
        )
        / NULLIF(
            cpf.total_purchase_lines,
            0
        ),
        4
    ) AS return_line_rate

FROM customer_purchase_features cpf

LEFT JOIN customer_return_features crf
    ON cpf.customer_id = crf.customer_id;