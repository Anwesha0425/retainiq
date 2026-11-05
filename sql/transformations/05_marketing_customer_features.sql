DROP VIEW IF EXISTS marketing_customer_features;

CREATE VIEW marketing_customer_features AS

SELECT
    customer_id,

    year_birth,
    education,
    marital_status,
    income,

    kidhome,
    teenhome,

    dt_customer,
    recency,

    mnt_wines,
    mnt_fruits,
    mnt_meat_products,
    mnt_fish_products,
    mnt_sweet_products,
    mnt_gold_prods,

    num_deals_purchases,
    num_web_purchases,
    num_catalog_purchases,
    num_store_purchases,
    num_web_visits_month,

    accepted_cmp1,
    accepted_cmp2,
    accepted_cmp3,
    accepted_cmp4,
    accepted_cmp5,

    complain,
    response,

    /* -------------------------
       Customer value
       ------------------------- */

    total_spend,

    total_purchases,

    total_campaign_acceptances,

    /* -------------------------
       Household
       ------------------------- */

    total_children,

    household_size,

    /* -------------------------
       Channel behavior
       ------------------------- */

    num_web_purchases
        + num_catalog_purchases
        + num_store_purchases
        AS total_channel_purchases,

    CASE
        WHEN (
            num_web_purchases
            + num_catalog_purchases
            + num_store_purchases
        ) > 0
        THEN
            num_web_purchases::NUMERIC
            / (
                num_web_purchases
                + num_catalog_purchases
                + num_store_purchases
            )
        ELSE 0
    END AS web_purchase_share,

    CASE
        WHEN (
            num_web_purchases
            + num_catalog_purchases
            + num_store_purchases
        ) > 0
        THEN
            num_catalog_purchases::NUMERIC
            / (
                num_web_purchases
                + num_catalog_purchases
                + num_store_purchases
            )
        ELSE 0
    END AS catalog_purchase_share,

    CASE
        WHEN (
            num_web_purchases
            + num_catalog_purchases
            + num_store_purchases
        ) > 0
        THEN
            num_store_purchases::NUMERIC
            / (
                num_web_purchases
                + num_catalog_purchases
                + num_store_purchases
            )
        ELSE 0
    END AS store_purchase_share,

    /* -------------------------
       Campaign engagement
       ------------------------- */

    CASE
        WHEN total_campaign_acceptances > 0
            THEN 1
        ELSE 0
    END AS accepted_any_campaign,

    /* -------------------------
       Customer age
       ------------------------- */

    approximate_age_2014,

    /* -------------------------
       Spending mix
       ------------------------- */

    CASE
        WHEN total_spend > 0
            THEN mnt_wines::NUMERIC / total_spend
        ELSE 0
    END AS wine_spend_share,

    CASE
        WHEN total_spend > 0
            THEN mnt_meat_products::NUMERIC / total_spend
        ELSE 0
    END AS meat_spend_share,

    CASE
        WHEN total_spend > 0
            THEN mnt_gold_prods::NUMERIC / total_spend
        ELSE 0
    END AS gold_spend_share

FROM stg_marketing_customers;