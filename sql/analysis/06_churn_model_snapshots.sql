DROP VIEW IF EXISTS churn_model_snapshots;

CREATE VIEW churn_model_snapshots AS

WITH parameters AS (
    SELECT
        DATE '2011-03-01' AS cutoff_date

    UNION ALL

    SELECT
        DATE '2011-05-01'

    UNION ALL

    SELECT
        DATE '2011-07-01'

    UNION ALL

    SELECT
        DATE '2011-09-01'
),

valid_transactions AS (
    SELECT
        customer_id,
        invoice,
        invoice_date,
        quantity,
        price,
        quantity * price AS line_amount,
        stock_code
    FROM stg_retail_transactions
    WHERE customer_id IS NOT NULL
      AND is_cancellation = FALSE
      AND quantity > 0
      AND price > 0
),

customer_features AS (
    SELECT
        p.cutoff_date,
        t.customer_id,

        COUNT(DISTINCT t.invoice) AS purchase_frequency,

        SUM(t.line_amount) AS total_spend,

        AVG(t.line_amount) AS average_line_value,

        COUNT(
            DISTINCT DATE_TRUNC(
                'month',
                t.invoice_date
            )
        ) AS active_purchase_months,

        COUNT(DISTINCT t.stock_code) AS unique_products_purchased,

        SUM(t.quantity) AS total_units_purchased,

        COUNT(*) AS total_purchase_lines,

        AVG(
            invoice_totals.order_value
        ) AS average_order_value

    FROM parameters p

    JOIN valid_transactions t
        ON t.invoice_date < p.cutoff_date

    JOIN (
        SELECT
            customer_id,
            invoice,
            SUM(line_amount) AS order_value
        FROM valid_transactions
        GROUP BY
            customer_id,
            invoice
    ) invoice_totals
        ON invoice_totals.customer_id = t.customer_id
       AND invoice_totals.invoice = t.invoice

    GROUP BY
        p.cutoff_date,
        t.customer_id
),

customer_returns AS (
    SELECT
        p.cutoff_date,
        t.customer_id,

        COUNT(*) AS return_lines,

        ABS(
            SUM(t.quantity)
        ) AS returned_units

    FROM parameters p

    JOIN stg_retail_transactions t
        ON t.invoice_date < p.cutoff_date

    WHERE t.customer_id IS NOT NULL
      AND t.quantity < 0

    GROUP BY
        p.cutoff_date,
        t.customer_id
),

future_purchases AS (
    SELECT DISTINCT
        p.cutoff_date,
        t.customer_id
    FROM parameters p

    JOIN valid_transactions t
        ON t.invoice_date >= p.cutoff_date
       AND t.invoice_date < (
            p.cutoff_date + INTERVAL '90 days'
       )
),

final_dataset AS (
    SELECT
        cf.cutoff_date,
        cf.customer_id,

        cf.purchase_frequency,
        cf.total_spend,
        cf.average_line_value,
        cf.active_purchase_months,
        cf.unique_products_purchased,
        cf.total_units_purchased,
        cf.total_purchase_lines,
        cf.average_order_value,

        COALESCE(
            cr.return_lines,
            0
        ) AS return_lines,

        COALESCE(
            cr.returned_units,
            0
        ) AS returned_units,

        CASE
            WHEN cf.total_purchase_lines = 0
                THEN 0
            ELSE
                COALESCE(
                    cr.return_lines,
                    0
                )::NUMERIC
                / cf.total_purchase_lines
        END AS return_line_rate,

        CASE
            WHEN fp.customer_id IS NULL
                THEN 1
            ELSE 0
        END AS churn

    FROM customer_features cf

    LEFT JOIN customer_returns cr
        ON cr.cutoff_date = cf.cutoff_date
       AND cr.customer_id = cf.customer_id

    LEFT JOIN future_purchases fp
        ON fp.cutoff_date = cf.cutoff_date
       AND fp.customer_id = cf.customer_id
)

SELECT *
FROM final_dataset;