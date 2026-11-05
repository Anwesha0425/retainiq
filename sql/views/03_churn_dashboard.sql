CREATE OR REPLACE VIEW churn_dashboard AS

SELECT
    m.customer_id,
    m.purchase_frequency,
    m.total_spend,
    m.average_line_value,
    m.active_purchase_months,
    m.unique_products_purchased,
    m.total_units_purchased,
    m.total_purchase_lines,
    m.first_purchase_date,
    m.last_purchase_date,
    m.return_lines,
    m.returned_units,
    m.average_order_value,
    m.return_line_rate,
    c.days_since_last_purchase,
    c.churn
FROM customer_behavior_features m
INNER JOIN customer_churn c
    ON m.customer_id = c.customer_id;