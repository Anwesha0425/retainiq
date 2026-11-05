import pandas as pd

from src.data.loader import (
    prepare_retail_dataframe,
    prepare_marketing_dataframe,
)


# =========================================================
# Retail loader tests
# =========================================================

def make_retail_dataframe():
    return pd.DataFrame({
        "Invoice": ["10001", "C10002"],
        "StockCode": ["A", "B"],
        "Description": ["Product A", "Product B"],
        "Quantity": [2, -1],
        "InvoiceDate": [
            "2010-01-01 10:30:00",
            "2010-01-02 11:45:00",
        ],
        "Price": [10.0, 5.0],
        "Customer ID": [1001.0, None],
        "Country": ["United Kingdom", "France"],
    })


def test_retail_column_mapping():
    df = make_retail_dataframe()

    result = prepare_retail_dataframe(df)

    expected_columns = [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "price",
        "customer_id",
        "country",
    ]

    assert list(result.columns) == expected_columns


def test_retail_row_count_preserved():
    df = make_retail_dataframe()

    result = prepare_retail_dataframe(df)

    assert len(result) == len(df)


def test_retail_customer_id_preserves_missing_values():
    df = make_retail_dataframe()

    result = prepare_retail_dataframe(df)

    assert result["customer_id"].isna().sum() == 1


def test_retail_customer_id_is_nullable_integer():
    df = make_retail_dataframe()

    result = prepare_retail_dataframe(df)

    assert str(result["customer_id"].dtype) == "Int64"


def test_retail_quantity_is_nullable_integer():
    df = make_retail_dataframe()

    result = prepare_retail_dataframe(df)

    assert str(result["quantity"].dtype) == "Int64"


def test_retail_invoice_date_is_datetime():
    df = make_retail_dataframe()

    result = prepare_retail_dataframe(df)

    assert pd.api.types.is_datetime64_any_dtype(
        result["invoice_date"]
    )


def test_retail_negative_quantity_is_preserved():
    df = make_retail_dataframe()

    result = prepare_retail_dataframe(df)

    assert result.loc[1, "quantity"] == -1


# =========================================================
# Marketing loader tests
# =========================================================

def make_marketing_dataframe():
    return pd.DataFrame({
        "ID": [1, 2],
        "Year_Birth": [1980, 1990],
        "Education": ["Graduation", "Master"],
        "Marital_Status": ["Single", "Married"],
        "Income": [50000.0, None],
        "Kidhome": [0, 1],
        "Teenhome": [0, 0],
        "Dt_Customer": ["01-01-2013", "02-01-2013"],
        "Recency": [10, 20],
        "MntWines": [100, 200],
        "MntFruits": [10, 20],
        "MntMeatProducts": [50, 100],
        "MntFishProducts": [20, 30],
        "MntSweetProducts": [10, 20],
        "MntGoldProds": [20, 30],
        "NumDealsPurchases": [2, 3],
        "NumWebPurchases": [5, 6],
        "NumCatalogPurchases": [2, 3],
        "NumStorePurchases": [5, 6],
        "NumWebVisitsMonth": [5, 6],
        "AcceptedCmp3": [0, 1],
        "AcceptedCmp4": [0, 0],
        "AcceptedCmp5": [0, 0],
        "AcceptedCmp1": [1, 0],
        "AcceptedCmp2": [0, 0],
        "Complain": [0, 0],
        "Z_CostContact": [3, 3],
        "Z_Revenue": [11, 11],
        "Response": [1, 0],
    })


def test_marketing_column_mapping():
    df = make_marketing_dataframe()

    result = prepare_marketing_dataframe(df)

    expected_columns = [
        "customer_id",
        "year_birth",
        "education",
        "marital_status",
        "income",
        "kidhome",
        "teenhome",
        "dt_customer",
        "recency",
        "mnt_wines",
        "mnt_fruits",
        "mnt_meat_products",
        "mnt_fish_products",
        "mnt_sweet_products",
        "mnt_gold_prods",
        "num_deals_purchases",
        "num_web_purchases",
        "num_catalog_purchases",
        "num_store_purchases",
        "num_web_visits_month",
        "accepted_cmp1",
        "accepted_cmp2",
        "accepted_cmp3",
        "accepted_cmp4",
        "accepted_cmp5",
        "complain",
        "z_cost_contact",
        "z_revenue",
        "response",
    ]

    assert list(result.columns) == expected_columns


def test_marketing_row_count_preserved():
    df = make_marketing_dataframe()

    result = prepare_marketing_dataframe(df)

    assert len(result) == len(df)


def test_marketing_customer_id_is_integer():
    df = make_marketing_dataframe()

    result = prepare_marketing_dataframe(df)

    assert str(result["customer_id"].dtype) == "Int64"


def test_marketing_missing_income_is_preserved():
    df = make_marketing_dataframe()

    result = prepare_marketing_dataframe(df)

    assert result["income"].isna().sum() == 1


def test_marketing_date_conversion():
    df = make_marketing_dataframe()

    result = prepare_marketing_dataframe(df)

    assert result.loc[0, "dt_customer"].year == 2013
    assert result.loc[0, "dt_customer"].month == 1
    assert result.loc[0, "dt_customer"].day == 1


def test_marketing_response_values_preserved():
    df = make_marketing_dataframe()

    result = prepare_marketing_dataframe(df)

    assert result["response"].tolist() == [1, 0]


def test_marketing_campaign_columns_preserved():
    df = make_marketing_dataframe()

    result = prepare_marketing_dataframe(df)

    assert result["accepted_cmp1"].tolist() == [1, 0]
    assert result["accepted_cmp2"].tolist() == [0, 0]
    assert result["accepted_cmp3"].tolist() == [0, 1]