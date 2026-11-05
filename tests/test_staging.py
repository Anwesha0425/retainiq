import pandas as pd


# =========================================================
# Retail staging logic
# =========================================================

def test_retail_line_amount():
    df = pd.DataFrame({
        "quantity": [2, -3],
        "price": [10.0, 5.0],
    })

    df["line_amount"] = (
        df["quantity"] * df["price"]
    )

    assert df.loc[0, "line_amount"] == 20.0
    assert df.loc[1, "line_amount"] == -15.0


def test_retail_cancellation_flag():
    df = pd.DataFrame({
        "invoice": [
            "10001",
            "C10002",
        ]
    })

    df["is_cancellation"] = (
        df["invoice"].str.startswith("C")
    )

    assert df["is_cancellation"].tolist() == [
        False,
        True,
    ]


def test_retail_return_flag():
    df = pd.DataFrame({
        "quantity": [5, -2, 0]
    })

    df["is_return"] = df["quantity"] < 0

    assert df["is_return"].tolist() == [
        False,
        True,
        False,
    ]


def test_retail_customer_flag():
    df = pd.DataFrame({
        "customer_id": [
            1001,
            None,
        ]
    })

    df["has_customer_id"] = (
        df["customer_id"].notna()
    )

    assert df["has_customer_id"].tolist() == [
        True,
        False,
    ]


# =========================================================
# Marketing staging logic
# =========================================================

def test_marketing_total_spend():
    df = pd.DataFrame({
        "mnt_wines": [100],
        "mnt_fruits": [20],
        "mnt_meat_products": [50],
        "mnt_fish_products": [10],
        "mnt_sweet_products": [20],
        "mnt_gold_prods": [30],
    })

    df["total_spend"] = (
        df["mnt_wines"]
        + df["mnt_fruits"]
        + df["mnt_meat_products"]
        + df["mnt_fish_products"]
        + df["mnt_sweet_products"]
        + df["mnt_gold_prods"]
    )

    assert df.loc[0, "total_spend"] == 230


def test_marketing_total_purchases():
    df = pd.DataFrame({
        "num_web_purchases": [5],
        "num_catalog_purchases": [3],
        "num_store_purchases": [7],
    })

    df["total_purchases"] = (
        df["num_web_purchases"]
        + df["num_catalog_purchases"]
        + df["num_store_purchases"]
    )

    assert df.loc[0, "total_purchases"] == 15


def test_marketing_campaign_acceptances():
    df = pd.DataFrame({
        "accepted_cmp1": [1],
        "accepted_cmp2": [0],
        "accepted_cmp3": [1],
        "accepted_cmp4": [0],
        "accepted_cmp5": [1],
    })

    df["total_campaign_acceptances"] = (
        df["accepted_cmp1"]
        + df["accepted_cmp2"]
        + df["accepted_cmp3"]
        + df["accepted_cmp4"]
        + df["accepted_cmp5"]
    )

    assert (
        df.loc[0, "total_campaign_acceptances"]
        == 3
    )


def test_marketing_household_size():
    df = pd.DataFrame({
        "kidhome": [1],
        "teenhome": [2],
    })

    df["household_size"] = (
        1
        + df["kidhome"]
        + df["teenhome"]
    )

    assert df.loc[0, "household_size"] == 4


def test_marketing_campaign_response():
    df = pd.DataFrame({
        "response": [1, 0]
    })

    df["responded"] = (
        df["response"] == 1
    )

    assert df["responded"].tolist() == [
        True,
        False,
    ]