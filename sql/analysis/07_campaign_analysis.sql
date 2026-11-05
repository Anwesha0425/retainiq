-- ============================================================
-- 07_campaign_analysis.sql
-- Marketing Campaign Analysis
-- ============================================================


-- ============================================================
-- 1. Overall Campaign Performance
-- ============================================================

DROP VIEW IF EXISTS campaign_analysis;

CREATE VIEW campaign_analysis AS

WITH campaign_summary AS (
    SELECT
        COUNT(*) AS total_customers,

        SUM(response) AS latest_campaign_responders,

        SUM(accepted_cmp1) AS campaign_1_acceptances,
        SUM(accepted_cmp2) AS campaign_2_acceptances,
        SUM(accepted_cmp3) AS campaign_3_acceptances,
        SUM(accepted_cmp4) AS campaign_4_acceptances,
        SUM(accepted_cmp5) AS campaign_5_acceptances,

        SUM(accepted_any_campaign)
            AS customers_accepting_any_campaign,

        SUM(complain)
            AS customers_with_complaints

    FROM marketing_customer_features
)

SELECT
    total_customers,

    latest_campaign_responders,

    ROUND(
        latest_campaign_responders::NUMERIC
        / NULLIF(total_customers, 0) * 100,
        2
    ) AS latest_campaign_response_rate,

    campaign_1_acceptances,
    campaign_2_acceptances,
    campaign_3_acceptances,
    campaign_4_acceptances,
    campaign_5_acceptances,

    customers_accepting_any_campaign,

    ROUND(
        customers_accepting_any_campaign::NUMERIC
        / NULLIF(total_customers, 0) * 100,
        2
    ) AS historical_campaign_engagement_rate,

    customers_with_complaints,

    ROUND(
        customers_with_complaints::NUMERIC
        / NULLIF(total_customers, 0) * 100,
        2
    ) AS complaint_rate

FROM campaign_summary;


-- ============================================================
-- 2. Campaign Response by Education
-- ============================================================

DROP VIEW IF EXISTS campaign_response_by_segment;

CREATE VIEW campaign_response_by_segment AS

SELECT
    education,

    COUNT(*) AS customers,

    SUM(response) AS responders,

    ROUND(
        AVG(response)::NUMERIC * 100,
        2
    ) AS response_rate,

    ROUND(
        AVG(total_spend)::NUMERIC,
        2
    ) AS average_spend,

    ROUND(
        AVG(income)::NUMERIC,
        2
    ) AS average_income,

    ROUND(
        AVG(total_purchases)::NUMERIC,
        2
    ) AS average_purchases,

    ROUND(
        AVG(num_web_visits_month)::NUMERIC,
        2
    ) AS average_web_visits,

    ROUND(
        AVG(total_campaign_acceptances)::NUMERIC,
        2
    ) AS average_campaign_acceptances

FROM marketing_customer_features

GROUP BY education;


-- ============================================================
-- 3. Campaign Response by Customer Value
-- ============================================================

DROP VIEW IF EXISTS campaign_response_by_value;

CREATE VIEW campaign_response_by_value AS

WITH customer_buckets AS (
    SELECT
        customer_id,
        response,
        total_spend,
        total_purchases,

        NTILE(4) OVER (
            ORDER BY total_spend
        ) AS spend_quartile

    FROM marketing_customer_features
)

SELECT
    spend_quartile,

    COUNT(*) AS customers,

    SUM(response) AS responders,

    ROUND(
        AVG(response)::NUMERIC * 100,
        2
    ) AS response_rate,

    ROUND(
        AVG(total_spend)::NUMERIC,
        2
    ) AS average_spend,

    ROUND(
        AVG(total_purchases)::NUMERIC,
        2
    ) AS average_purchases

FROM customer_buckets

GROUP BY spend_quartile

ORDER BY spend_quartile;