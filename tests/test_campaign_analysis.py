import pandas as pd
import pytest

from src.statistics.campaign_analysis import (
    chi_square_response_test,
    compare_response_groups,
    summarize_significant_features,
)


def make_sample_data():
    return pd.DataFrame(
        {
            "education": [
                "Basic",
                "Basic",
                "Master",
                "Master",
                "PhD",
                "PhD",
                "Graduation",
                "Graduation",
                "Graduation",
                "Graduation",
            ],
            "response": [
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
            ],
            "total_spend": [
                100,
                150,
                400,
                700,
                900,
                1200,
                300,
                800,
                250,
                600,
            ],
            "total_purchases": [
                2,
                3,
                5,
                8,
                9,
                12,
                4,
                10,
                3,
                7,
            ],
            "income": [
                20000,
                22000,
                40000,
                50000,
                60000,
                70000,
                35000,
                65000,
                30000,
                55000,
            ],
        }
    )


def test_chi_square_response_test():
    df = make_sample_data()

    result = chi_square_response_test(
        df,
        "education",
    )

    assert result["feature"] == "education"
    assert "chi2" in result
    assert "p_value" in result
    assert "degrees_of_freedom" in result

    assert result["chi2"] >= 0
    assert 0 <= result["p_value"] <= 1


def test_chi_square_rejects_missing_column():
    df = make_sample_data()

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        chi_square_response_test(
            df,
            "missing_column",
        )


def test_compare_response_groups():
    df = make_sample_data()

    results = compare_response_groups(
        df,
        [
            "total_spend",
            "total_purchases",
            "income",
        ],
    )

    assert len(results) == 3

    assert set(
        [
            "feature",
            "responder_median",
            "non_responder_median",
            "u_statistic",
            "p_value",
            "effect_size",
            "adjusted_p_value",
            "significant",
        ]
    ).issubset(results.columns)


def test_adjusted_p_values_are_valid():
    df = make_sample_data()

    results = compare_response_groups(
        df,
        [
            "total_spend",
            "total_purchases",
            "income",
        ],
    )

    assert results[
        "adjusted_p_value"
    ].between(0, 1).all()


def test_significance_column_is_boolean():
    df = make_sample_data()

    results = compare_response_groups(
        df,
        [
            "total_spend",
            "total_purchases",
        ],
    )

    assert results["significant"].dtype == bool


def test_summarize_significant_features():
    results = pd.DataFrame(
        {
            "feature": [
                "feature_a",
                "feature_b",
                "feature_c",
            ],
            "adjusted_p_value": [
                0.001,
                0.20,
                0.03,
            ],
            "significant": [
                True,
                False,
                True,
            ],
        }
    )

    significant = summarize_significant_features(
        results
    )

    assert len(significant) == 2
    assert set(
        significant["feature"]
    ) == {
        "feature_a",
        "feature_c",
    }


def test_summarize_rejects_missing_p_value_column():
    results = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "p_value": [0.01, 0.02],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing column",
    ):
        summarize_significant_features(
            results
        )


def test_no_missing_required_columns_for_comparison():
    df = make_sample_data()

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        compare_response_groups(
            df,
            [
                "total_spend",
                "unknown_feature",
            ],
        )


def test_response_groups_are_binary():
    df = make_sample_data()

    assert set(
        df["response"].unique()
    ).issubset({0, 1})


def test_result_contains_all_requested_features():
    df = make_sample_data()

    features = [
        "total_spend",
        "total_purchases",
        "income",
    ]

    results = compare_response_groups(
        df,
        features,
    )

    assert list(results["feature"]) == features

def test_effect_size_is_non_negative():
    df = make_sample_data()

    results = compare_response_groups(
        df,
        [
            "total_spend",
            "total_purchases",
            "income",
        ],
    )

    assert (results["effect_size"] >= 0).all()


def test_effect_size_is_numeric():
    df = make_sample_data()

    results = compare_response_groups(
        df,
        ["total_spend"],
    )

    assert pd.api.types.is_numeric_dtype(
        results["effect_size"]
    )