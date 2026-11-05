-- =========================================================
-- Customer & Marketing Analytics
-- Raw PostgreSQL Schema
-- =========================================================


-- =========================================================
-- Online Retail II
-- Grain: one row per invoice line
-- =========================================================

CREATE TABLE IF NOT EXISTS retail_transactions (
    retail_transaction_id BIGSERIAL PRIMARY KEY,

    invoice VARCHAR(20) NOT NULL,
    stock_code VARCHAR(50) NOT NULL,
    description TEXT,

    quantity INTEGER NOT NULL,

    invoice_date TIMESTAMP NOT NULL,

    price NUMERIC(12, 4) NOT NULL,

    customer_id INTEGER,

    country VARCHAR(100)
);


-- =========================================================
-- Customer Personality / Marketing Campaign
-- Grain: one row per customer
-- =========================================================

CREATE TABLE IF NOT EXISTS marketing_customers (
    customer_id INTEGER PRIMARY KEY,

    year_birth INTEGER NOT NULL,

    education VARCHAR(50),
    marital_status VARCHAR(50),

    income NUMERIC(12, 2),

    kidhome INTEGER,
    teenhome INTEGER,

    dt_customer DATE,

    recency INTEGER,

    mnt_wines INTEGER,
    mnt_fruits INTEGER,
    mnt_meat_products INTEGER,
    mnt_fish_products INTEGER,
    mnt_sweet_products INTEGER,
    mnt_gold_prods INTEGER,

    num_deals_purchases INTEGER,
    num_web_purchases INTEGER,
    num_catalog_purchases INTEGER,
    num_store_purchases INTEGER,
    num_web_visits_month INTEGER,

    accepted_cmp1 SMALLINT,
    accepted_cmp2 SMALLINT,
    accepted_cmp3 SMALLINT,
    accepted_cmp4 SMALLINT,
    accepted_cmp5 SMALLINT,

    complain SMALLINT,

    z_cost_contact INTEGER,
    z_revenue INTEGER,

    response SMALLINT
);