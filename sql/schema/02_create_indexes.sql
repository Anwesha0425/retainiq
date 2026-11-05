-- =========================================================
-- Retail transaction indexes
-- =========================================================

CREATE INDEX IF NOT EXISTS idx_retail_invoice
    ON retail_transactions(invoice);

CREATE INDEX IF NOT EXISTS idx_retail_invoice_date
    ON retail_transactions(invoice_date);

CREATE INDEX IF NOT EXISTS idx_retail_customer_id
    ON retail_transactions(customer_id);

CREATE INDEX IF NOT EXISTS idx_retail_stock_code
    ON retail_transactions(stock_code);

CREATE INDEX IF NOT EXISTS idx_retail_country
    ON retail_transactions(country);


-- =========================================================
-- Marketing customer indexes
-- =========================================================

CREATE INDEX IF NOT EXISTS idx_marketing_response
    ON marketing_customers(response);

CREATE INDEX IF NOT EXISTS idx_marketing_income
    ON marketing_customers(income);

CREATE INDEX IF NOT EXISTS idx_marketing_education
    ON marketing_customers(education);

CREATE INDEX IF NOT EXISTS idx_marketing_dt_customer
    ON marketing_customers(dt_customer);