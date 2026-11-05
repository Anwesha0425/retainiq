import pandas as pd
import pytest


REQUIRED_COLUMNS = {
    "customer_id",
    "total_spend",
    "total_purchases",
    "total_campaign_acceptances",
    "total_children",
    "household_size",
    "total_channel_purchases",
    "web_purchase_share",
    "catalog_purchase_share",
    "store_purchase_share",
    "accepted_any_campaign",
    "approximate_age_2014",
    "wine_spend_share",
    "meat_spend_share",
    "gold_spend_share",
}


def make_sample_data():
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "total_spend": [1000.0, 500.0, 0.0],
            "total_purchases": [20, 10, 0],
            "total_campaign_acceptances": [2, 0, 1],
            "total_children": [1, 2, 0],
            "household_size": [3, 4, 2],
            "total_channel_purchases": [20, 10, 0],
            "web_purchase_share": [0.5, 0.2, 0.0],
            "catalog_purchase_share": [0.3, 0.5, 0.0],
            "store_purchase_share": [0.2, 0.3, 0.0],
            "accepted_any_campaign": [1, 0, 1],
            "approximate_age_2014": [40, 50, 30],
            "wine_spend_share": [0.5, 0.4, 0.0],
            "meat_spend_share": [0.3, 0.2, 0.0],
            "gold_spend_share": [0.1, 0.2, 0.0],
        }
    )


def test_required_columns_exist():
    df = make_sample_data()

    assert REQUIRED_COLUMNS.issubset(df.columns)


def test_one_row_per_customer():
    df = make_sample_data()

    assert df["customer_id"].is_unique


def test_total_channel_purchases_is_non_negative():
    df = make_sample_data()

    assert (df["total_channel_purchases"] >= 0).all()


def test_channel_shares_are_between_zero_and_one():
    df = make_sample_data()

    share_columns = [
        "web_purchase_share",
        "catalog_purchase_share",
        "store_purchase_share",
    ]

    for column in share_columns:
        assert df[column].between(0, 1).all()


def test_spending_shares_are_between_zero_and_one():
    df = make_sample_data()

    share_columns = [
        "wine_spend_share",
        "meat_spend_share",
        "gold_spend_share",
    ]

    for column in share_columns:
        assert df[column].between(0, 1).all()


def test_campaign_flag_is_binary():
    df = make_sample_data()

    assert set(
        df["accepted_any_campaign"].unique()
    ).issubset({0, 1})


def test_customer_count_matches_unique_ids():
    df = make_sample_data()

    assert len(df) == df["customer_id"].nunique()


def test_zero_purchase_customer_has_zero_channel_shares():
    df = make_sample_data()

    zero_purchase = df[
        df["total_channel_purchases"] == 0
    ].iloc[0]

    assert zero_purchase["web_purchase_share"] == 0
    assert zero_purchase["catalog_purchase_share"] == 0
    assert zero_purchase["store_purchase_share"] == 0


def test_zero_spend_customer_has_zero_spending_shares():
    df = make_sample_data()

    zero_spend = df[
        df["total_spend"] == 0
    ].iloc[0]

    assert zero_spend["wine_spend_share"] == 0
    assert zero_spend["meat_spend_share"] == 0
    assert zero_spend["gold_spend_share"] == 0