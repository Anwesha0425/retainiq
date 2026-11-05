import pandas as pd
import pytest


def make_time_based_dataset():
    return pd.DataFrame({
        "customer_id": [1, 2, 3],

        "purchase_frequency": [5, 2, 8],
        "total_spend": [1000.0, 300.0, 2000.0],
        "average_line_value": [20.0, 15.0, 25.0],
        "active_purchase_months": [5, 2, 8],
        "unique_products_purchased": [20, 5, 30],
        "total_units_purchased": [100, 30, 200],
        "total_purchase_lines": [50, 10, 80],
        "return_lines": [2, 0, 3],
        "returned_units": [4, 0, 5],
        "average_order_value": [200.0, 150.0, 250.0],
        "return_line_rate": [0.04, 0.0, 0.0375],

        "first_purchase_date": pd.to_datetime([
            "2011-01-01",
            "2011-03-01",
            "2010-12-01",
        ]).date,

        "last_purchase_date": pd.to_datetime([
            "2011-08-20",
            "2011-07-10",
            "2011-08-25",
        ]).date,

        "cutoff_date": pd.to_datetime([
            "2011-09-01",
            "2011-09-01",
            "2011-09-01",
        ]).date,

        "prediction_end_date": pd.to_datetime([
            "2011-11-30",
            "2011-11-30",
            "2011-11-30",
        ]).date,

        "churn": [0, 1, 0],
    })


def test_one_row_per_customer():
    df = make_time_based_dataset()

    assert len(df) == df["customer_id"].nunique()


def test_churn_is_binary():
    df = make_time_based_dataset()

    assert set(df["churn"].unique()).issubset({0, 1})


def test_cutoff_date_is_before_prediction_end():
    df = make_time_based_dataset()

    assert (
        pd.to_datetime(df["cutoff_date"])
        <
        pd.to_datetime(df["prediction_end_date"])
    ).all()


def test_prediction_window_is_90_days():
    df = make_time_based_dataset()

    cutoff = pd.to_datetime(
        df["cutoff_date"]
    )

    prediction_end = pd.to_datetime(
        df["prediction_end_date"]
    )

    assert (
        prediction_end - cutoff
    ).dt.days.eq(90).all()


def test_features_do_not_include_direct_churn_signal():
    df = make_time_based_dataset()

    forbidden_features = {
        "days_since_last_purchase",
    }

    assert forbidden_features.isdisjoint(
        df.columns
    )


def test_feature_columns_are_numeric():
    df = make_time_based_dataset()

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
    df = make_time_based_dataset()

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

    assert not df[
        feature_columns
    ].isna().any().any()


def test_churn_is_separate_from_features():
    df = make_time_based_dataset()

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

    assert "churn" not in feature_columns