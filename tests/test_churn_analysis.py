import pandas as pd
import pytest

from src.statistics.churn_analysis import (
    compare_churn_groups,
    summarize_significant_features,
)


def make_test_data():
    return pd.DataFrame({
        "total_spend": [
            100,
            120,
            110,
            130,
            1000,
            1200,
            1100,
            1300,
        ],
        "purchase_frequency": [
            2,
            3,
            2,
            3,
            10,
            12,
            11,
            13,
        ],
        "churn": [
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
        ],
    })


def test_compare_churn_groups_returns_dataframe():
    df = make_test_data()

    result = compare_churn_groups(
        df,
        [
            "total_spend",
            "purchase_frequency",
        ],
    )

    assert isinstance(result, pd.DataFrame)


def test_compare_churn_groups_has_one_row_per_feature():
    df = make_test_data()

    features = [
        "total_spend",
        "purchase_frequency",
    ]

    result = compare_churn_groups(
        df,
        features,
    )

    assert len(result) == len(features)


def test_result_contains_expected_columns():
    df = make_test_data()

    result = compare_churn_groups(
        df,
        ["total_spend"],
    )

    expected_columns = {
        "feature",
        "non_churned_median",
        "churned_median",
        "u_statistic",
        "p_value",
        "effect_size",
        "adjusted_p_value",
        "significant",
    }

    assert expected_columns.issubset(
        set(result.columns)
    )


def test_p_values_are_valid():
    df = make_test_data()

    result = compare_churn_groups(
        df,
        ["total_spend"],
    )

    assert (
        (result["p_value"] >= 0)
        & (result["p_value"] <= 1)
    ).all()


def test_adjusted_p_values_are_valid():
    df = make_test_data()

    result = compare_churn_groups(
        df,
        ["total_spend", "purchase_frequency"],
    )

    assert (
        (result["adjusted_p_value"] >= 0)
        & (result["adjusted_p_value"] <= 1)
    ).all()


def test_effect_size_is_non_negative():
    df = make_test_data()

    result = compare_churn_groups(
        df,
        ["total_spend"],
    )

    assert (
        result["effect_size"] >= 0
    ).all()


def test_invalid_target_values_raise_error():
    df = make_test_data()

    df.loc[0, "churn"] = 2

    with pytest.raises(ValueError):
        compare_churn_groups(
            df,
            ["total_spend"],
        )


def test_missing_feature_raises_error():
    df = make_test_data()

    with pytest.raises(ValueError):
        compare_churn_groups(
            df,
            ["missing_feature"],
        )


def test_non_numeric_feature_raises_error():
    df = make_test_data()

    df["segment"] = [
        "A",
        "A",
        "B",
        "B",
        "A",
        "A",
        "B",
        "B",
    ]

    with pytest.raises(ValueError):
        compare_churn_groups(
            df,
            ["segment"],
        )


def test_significant_feature_summary():
    results = pd.DataFrame({
        "feature": [
            "feature_a",
            "feature_b",
        ],
        "adjusted_p_value": [
            0.001,
            0.80,
        ],
        "significant": [
            True,
            False,
        ],
    })

    result = summarize_significant_features(
        results
    )

    assert len(result) == 1
    assert result.iloc[0]["feature"] == "feature_a"