-- =========================================================
-- RFM Segmentation
-- Grain: one row per customer
-- =========================================================

DROP VIEW IF EXISTS customer_rfm_segments;

CREATE VIEW customer_rfm_segments AS

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

    (
        recency_score
        + frequency_score
        + monetary_score
    ) AS rfm_score,

    CONCAT(
        recency_score,
        frequency_score,
        monetary_score
    ) AS rfm_code,

    CASE

        WHEN recency_score >= 4
             AND frequency_score >= 4
             AND monetary_score >= 4
        THEN 'Champions'

        WHEN recency_score >= 4
             AND frequency_score >= 3
        THEN 'Loyal Customers'

        WHEN recency_score >= 4
             AND frequency_score <= 2
        THEN 'Recent Customers'

        WHEN recency_score <= 2
             AND frequency_score >= 4
        THEN 'At Risk'

        WHEN recency_score <= 2
             AND monetary_score >= 4
        THEN 'High Value At Risk'

        WHEN recency_score <= 2
             AND frequency_score <= 2
        THEN 'Lost Customers'

        ELSE 'Potential Loyalists'

    END AS customer_segment

FROM customer_rfm_scores;