-- =========================================================
-- RFM Scoring
-- Grain: one row per customer
-- =========================================================

DROP VIEW IF EXISTS customer_rfm_scores;

CREATE VIEW customer_rfm_scores AS

SELECT
    customer_id,

    last_purchase_date,
    analysis_date,

    recency,
    frequency,
    monetary,

    -- Recency:
    -- lower number of days is better
    NTILE(5) OVER (
        ORDER BY recency DESC
    ) AS recency_score,

    -- Frequency:
    -- higher frequency is better
    NTILE(5) OVER (
        ORDER BY frequency
    ) AS frequency_score,

    -- Monetary:
    -- higher spending is better
    NTILE(5) OVER (
        ORDER BY monetary
    ) AS monetary_score

FROM customer_rfm;