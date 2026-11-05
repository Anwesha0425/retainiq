import pandas as pd


# =========================================================
# Test modeling dataset
# =========================================================

def make_model_dataset():
    return pd.DataFrame({
        "customer_id": [1, 2, 3],

        "purchase_frequency": [5, 2, 10],
        "total_spend": [500.0, 100.0, 2000.0],
        "average_line_value": [50.0, 25.0, 100.0],
        "active_purchase_months": [4, 2, 8],
        "unique_products_purchased": [10, 3, 25],
        "total_units_purchased": [50, 10, 200],
        "total_purchase_lines": [20, 5, 50],
        "return_lines": [1, 0, 2],
        "returned_units": [2, 0, 5],
        "average_order_value": [100.0, 50.0, 200.0],
        "return_line_rate": [0.05, 0.0, 0.04],

        "churn": [0, 1, 0],
    })


def test_model_dataset_has_one_row_per_customer():
    df = make_model_dataset()

    assert len(df) == df["customer_id"].nunique()


def test_churn_is_binary():
    df = make_model_dataset()

    assert set(df["churn"].unique()).issubset({0, 1})


def test_model_features_are_numeric():
    df = make_model_dataset()

    feature_columns = [
        "purchase_frequency",
        "total_spend",
        "average_line_value",
        "active_purchase_months",
        "unique_products_purchased",
        "total_units_purchased",
        "total_purchase_lines",
        "return_lines",
        "returned_units",
        "average_order_value",
        "return_line_rate",
    ]

    for column in feature_columns:
        assert pd.api.types.is_numeric_dtype(
            df[column]
        )


def test_model_features_have_no_missing_values():
    df = make_model_dataset()

    feature_columns = [
        "purchase_frequency",
        "total_spend",
        "average_line_value",
        "active_purchase_months",
        "unique_products_purchased",
        "total_units_purchased",
        "total_purchase_lines",
        "return_lines",
        "returned_units",
        "average_order_value",
        "return_line_rate",
    ]

    assert not df[feature_columns].isna().any().any()


def test_days_since_last_purchase_not_in_features():
    df = make_model_dataset()

    assert "days_since_last_purchase" not in df.columns


def test_last_purchase_date_not_in_features():
    df = make_model_dataset()

    assert "last_purchase_date" not in df.columns


def test_churn_is_not_used_as_feature():
    df = make_model_dataset()

    feature_columns = [
        column
        for column in df.columns
        if column not in ["customer_id", "churn"]
    ]

    assert "churn" not in feature_columns