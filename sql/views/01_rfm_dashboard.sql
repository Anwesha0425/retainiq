CREATE OR REPLACE VIEW rfm_dashboard AS

SELECT
    customer_id,
    last_purchase_date,
    analysis_date,
    recency,
    frequency,
    monetary,
    recency_score,
    frequency_score,
    monetary_score,
    rfm_score,
    rfm_code,
    customer_segment
FROM customer_rfm_segments;