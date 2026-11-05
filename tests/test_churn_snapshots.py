import pandas as pd
import pytest


REQUIRED_COLUMNS = {
    "cutoff_date",
    "customer_id",
    "purchase_frequency",
    "total_spend",
    "average_line_value",
    "active_purchase_months",
    "unique_products_purchased",
    "total_units_purchased",
    "total_purchase_lines",
    "average_order_value",
    "return_lines",
    "returned_units",
    "return_line_rate",
    "churn",
}


def make_snapshot_data():
    return pd.DataFrame(
        {
            "cutoff_date": pd.to_datetime(
                [
                    "2011-03-01",
                    "2011-03-01",
                    "2011-05-01",
                    "2011-05-01",
                    "2011-07-01",
                    "2011-07-01",
                    "2011-09-01",
                    "2011-09-01",
                ]
            ),
            "customer_id": range(1, 9),
            "purchase_frequency": [5, 2, 6, 1, 7, 2, 8, 1],
            "total_spend": [500, 200, 600, 100, 700, 150, 800, 120],
            "average_line_value": [20, 15, 21, 10, 22, 12, 23, 11],
            "active_purchase_months": [4, 1, 5, 1, 6, 2, 7, 1],
            "unique_products_purchased": [20, 8, 25, 5, 30, 7, 35, 4],
            "total_units_purchased": [100, 30, 120, 20, 140, 25, 160, 15],
            "total_purchase_lines": [25, 8, 30, 5, 35, 7, 40, 4],
            "average_order_value": [100, 100, 100, 100, 100, 100, 100, 100],
            "return_lines": [1, 0, 1, 0, 2, 0, 1, 0],
            "returned_units": [2, 0, 2, 0, 3, 0, 2, 0],
            "return_line_rate": [0.04, 0, 0.03, 0, 0.057, 0, 0.025, 0],
            "churn": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_required_columns_exist():
    df = make_snapshot_data()

    assert REQUIRED_COLUMNS.issubset(df.columns)


def test_multiple_cutoff_dates_exist():
    df = make_snapshot_data()

    assert df["cutoff_date"].nunique() == 4


def test_cutoff_dates_are_chronological():
    df = make_snapshot_data()

    dates = sorted(df["cutoff_date"].unique())

    assert dates == sorted(dates)


def test_churn_is_binary():
    df = make_snapshot_data()

    assert set(df["churn"].unique()).issubset({0, 1})


def test_each_customer_snapshot_has_one_row():
    df = make_snapshot_data()

    duplicate_count = df.duplicated(
        subset=["cutoff_date", "customer_id"]
    ).sum()

    assert duplicate_count == 0


def test_training_cutoffs_are_before_final_test_cutoff():
    df = make_snapshot_data()

    final_cutoff = df["cutoff_date"].max()

    training_cutoffs = df.loc[
        df["cutoff_date"] < final_cutoff,
        "cutoff_date",
    ]

    assert len(training_cutoffs) > 0
    assert training_cutoffs.max() < final_cutoff


def test_final_cutoff_exists():
    df = make_snapshot_data()

    assert pd.Timestamp("2011-09-01") in set(
        df["cutoff_date"]
    )


def test_snapshot_features_do_not_include_target():
    feature_columns = {
        "purchase_frequency",
        "total_spend",
        "average_line_value",
        "active_purchase_months",
        "unique_products_purchased",
        "total_units_purchased",
        "total_purchase_lines",
        "average_order_value",
        "return_lines",
        "returned_units",
        "return_line_rate",
    }

    assert "churn" not in feature_columns