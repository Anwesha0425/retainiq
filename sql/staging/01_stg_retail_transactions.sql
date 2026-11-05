-- =========================================================
-- Staging: Online Retail II
-- Grain: one row per raw retail transaction
-- =========================================================

DROP VIEW IF EXISTS stg_retail_transactions;

CREATE VIEW stg_retail_transactions AS

SELECT
    retail_transaction_id,

    invoice,
    stock_code,
    description,

    quantity,
    invoice_date,

    price,

    customer_id,
    country,

    -- Monetary value of the transaction line
    quantity * price AS line_amount,

    -- Transaction classification
    CASE
        WHEN invoice LIKE 'C%' THEN TRUE
        ELSE FALSE
    END AS is_cancellation,

    CASE
        WHEN quantity < 0 THEN TRUE
        ELSE FALSE
    END AS is_return,

    CASE
        WHEN customer_id IS NOT NULL THEN TRUE
        ELSE FALSE
    END AS has_customer_id,

    CASE
        WHEN price = 0 THEN TRUE
        ELSE FALSE
    END AS is_zero_price,

    -- Date dimensions
    EXTRACT(YEAR FROM invoice_date)::INTEGER
        AS invoice_year,

    EXTRACT(MONTH FROM invoice_date)::INTEGER
        AS invoice_month,

    DATE_TRUNC(
        'month',
        invoice_date
    )::DATE AS invoice_month_start,

    TO_CHAR(
        invoice_date,
        'YYYY-MM'
    ) AS invoice_year_month

FROM retail_transactions;