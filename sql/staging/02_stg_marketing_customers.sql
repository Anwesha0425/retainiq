-- =========================================================
-- Staging: Customer Personality Analysis
-- Grain: one row per customer
-- =========================================================

DROP VIEW IF EXISTS stg_marketing_customers;

CREATE VIEW stg_marketing_customers AS

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

    z_cost_contact,
    z_revenue,

    response,

    -- Total customer spending
    COALESCE(mnt_wines, 0)
        + COALESCE(mnt_fruits, 0)
        + COALESCE(mnt_meat_products, 0)
        + COALESCE(mnt_fish_products, 0)
        + COALESCE(mnt_sweet_products, 0)
        + COALESCE(mnt_gold_prods, 0)
        AS total_spend,

    -- Total purchases across channels
    COALESCE(num_web_purchases, 0)
        + COALESCE(num_catalog_purchases, 0)
        + COALESCE(num_store_purchases, 0)
        AS total_purchases,

    -- Number of previous campaigns accepted
    COALESCE(accepted_cmp1, 0)
        + COALESCE(accepted_cmp2, 0)
        + COALESCE(accepted_cmp3, 0)
        + COALESCE(accepted_cmp4, 0)
        + COALESCE(accepted_cmp5, 0)
        AS total_campaign_acceptances,

    -- Household children
    COALESCE(kidhome, 0)
        + COALESCE(teenhome, 0)
        AS total_children,

    -- Children + teenagers + customer
    1
        + COALESCE(kidhome, 0)
        + COALESCE(teenhome, 0)
        AS household_size,

    -- Explicit campaign response flag
    CASE
        WHEN response = 1 THEN TRUE
        ELSE FALSE
    END AS responded_to_latest_campaign,

    -- Customer age at the approximate campaign period.
    -- The source contains birth year only, so this is
    -- intentionally based on birth year rather than an
    -- exact age.
    CASE
        WHEN year_birth IS NOT NULL
        THEN 2014 - year_birth
        ELSE NULL
    END AS approximate_age_2014

FROM marketing_customers;