-- =========================================================
-- Churn Modeling Dataset
-- Grain: one row per customer
-- =========================================================

DROP VIEW IF EXISTS churn_model_dataset;

CREATE VIEW churn_model_dataset AS

SELECT
    cbf.customer_id,

    -- Behavioral features
    cbf.purchase_frequency,
    cbf.total_spend,
    cbf.average_line_value,
    cbf.active_purchase_months,
    cbf.unique_products_purchased,
    cbf.total_units_purchased,
    cbf.total_purchase_lines,
    cbf.return_lines,
    cbf.returned_units,
    cbf.average_order_value,
    cbf.return_line_rate,

    -- Target
    cc.churn

FROM customer_behavior_features cbf

INNER JOIN customer_churn cc
    ON cbf.customer_id = cc.customer_id;