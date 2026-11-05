import pandas as pd

from src.data.validator import (
    validate_retail_data,
    validate_marketing_data,
)


# ---------------------------------------------------------
# Test data factories
# ---------------------------------------------------------

def make_retail_dataframe():
    return pd.DataFrame({
        "Invoice": ["10001", "10002", "C10003"],
        "StockCode": ["A", "B", "C"],
        "Description": [
            "Product A",
            "Product B",
            "Product C",
        ],
        "Quantity": [2, 3, -1],
        "InvoiceDate": [
            "2010-01-01 10:30:00",
            "2010-01-02 11:45:00",
            "2010-01-03 09:15:00",
        ],
        "Price": [10.0, 20.0, 5.0],
        "Customer ID": [1001, 1002, None],
        "Country": [
            "United Kingdom",
            "Germany",
            "France",
        ],
    })


def make_marketing_dataframe():
    return pd.DataFrame({
        "ID": [1, 2, 3],
        "Year_Birth": [1980, 1990, 1975],
        "Education": [
            "Graduation",
            "Master",
            "PhD",
        ],
        "Marital_Status": [
            "Single",
            "Married",
            "Together",
        ],
        "Income": [50000.0, 60000.0, 70000.0],
        "Kidhome": [0, 1, 0],
        "Teenhome": [0, 0, 1],
        "Dt_Customer": [
            "01-01-2013",
            "02-01-2013",
            "03-01-2013",
        ],
        "Recency": [10, 20, 30],
        "MntWines": [100, 200, 300],
        "MntFruits": [10, 20, 30],
        "MntMeatProducts": [50, 100, 150],
        "MntFishProducts": [20, 30, 40],
        "MntSweetProducts": [10, 20, 30],
        "MntGoldProds": [20, 30, 40],
        "NumDealsPurchases": [2, 3, 4],
        "NumWebPurchases": [5, 6, 7],
        "NumCatalogPurchases": [2, 3, 4],
        "NumStorePurchases": [5, 6, 7],
        "NumWebVisitsMonth": [5, 6, 7],
        "AcceptedCmp3": [0, 1, 0],
        "AcceptedCmp4": [0, 0, 1],
        "AcceptedCmp5": [0, 0, 0],
        "AcceptedCmp1": [1, 0, 0],
        "AcceptedCmp2": [0, 0, 1],
        "Complain": [0, 0, 0],
        "Z_CostContact": [3, 3, 3],
        "Z_Revenue": [11, 11, 11],
        "Response": [1, 0, 1],
    })


# ---------------------------------------------------------
# Retail validation tests
# ---------------------------------------------------------

def test_retail_required_columns():
    df = make_retail_dataframe()

    result = validate_retail_data(df)

    assert result["required_columns"] is True


def test_retail_row_count():
    df = make_retail_dataframe()

    result = validate_retail_data(df)

    assert result["row_count"] == 3


def test_retail_duplicate_detection():
    df = make_retail_dataframe()

    df = pd.concat(
        [df, df.iloc[[0]]],
        ignore_index=True,
    )

    result = validate_retail_data(df)

    assert result["duplicate_rows"] == 1


def test_retail_missing_customer_detection():
    df = make_retail_dataframe()

    result = validate_retail_data(df)

    assert result["missing_customer_ids"] == 1


def test_retail_negative_quantity_detection():
    df = make_retail_dataframe()

    result = validate_retail_data(df)

    assert result["negative_quantities"] == 1


def test_retail_negative_price_detection():
    df = make_retail_dataframe()

    df.loc[0, "Price"] = -10

    result = validate_retail_data(df)

    assert result["negative_prices"] == 1


def test_retail_zero_price_detection():
    df = make_retail_dataframe()

    df.loc[0, "Price"] = 0

    result = validate_retail_data(df)

    assert result["zero_prices"] == 1


def test_retail_invalid_date_detection():
    df = make_retail_dataframe()

    df.loc[0, "InvoiceDate"] = "invalid-date"

    result = validate_retail_data(df)

    assert result["invalid_invoice_dates"] == 1


# ---------------------------------------------------------
# Marketing validation tests
# ---------------------------------------------------------

def test_marketing_required_columns():
    df = make_marketing_dataframe()

    result = validate_marketing_data(df)

    assert result["required_columns"] is True


def test_marketing_row_count():
    df = make_marketing_dataframe()

    result = validate_marketing_data(df)

    assert result["row_count"] == 3


def test_marketing_duplicate_customer_detection():
    df = make_marketing_dataframe()

    df.loc[2, "ID"] = 1

    result = validate_marketing_data(df)

    assert result["duplicate_customer_ids"] == 1


def test_marketing_missing_income_detection():
    df = make_marketing_dataframe()

    df.loc[0, "Income"] = None

    result = validate_marketing_data(df)

    assert result["missing_income"] == 1


def test_marketing_response_validation():
    df = make_marketing_dataframe()

    result = validate_marketing_data(df)

    assert result["invalid_response_values"] == 0


def test_marketing_invalid_response_detection():
    df = make_marketing_dataframe()

    df.loc[0, "Response"] = 2

    result = validate_marketing_data(df)

    assert result["invalid_response_values"] == 1


def test_marketing_campaign_validation():
    df = make_marketing_dataframe()

    result = validate_marketing_data(df)

    assert result["invalid_campaign_values"] == 0


def test_marketing_invalid_campaign_detection():
    df = make_marketing_dataframe()

    df.loc[0, "AcceptedCmp1"] = 2

    result = validate_marketing_data(df)

    assert result["AcceptedCmp1_invalid"] == 1