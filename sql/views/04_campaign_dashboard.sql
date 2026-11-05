CREATE OR REPLACE VIEW campaign_dashboard AS

SELECT
    customer_id,
    education,
    marital_status,
    income,
    recency,
    total_spend,
    total_purchases,
    total_children,
    household_size,
    num_deals_purchases,
    num_web_purchases,
    num_catalog_purchases,
    num_store_purchases,
    num_web_visits_month,
    total_campaign_acceptances,
    web_purchase_share,
    catalog_purchase_share,
    store_purchase_share,
    accepted_any_campaign,
    response,
    approximate_age_2014
FROM marketing_customer_features;