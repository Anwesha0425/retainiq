import numpy as np
import pandas as pd
import pytest

from src.modeling.campaign_model import (
    DEFAULT_CAMPAIGN_FEATURES,
    build_campaign_logistic_regression,
    build_campaign_random_forest,
    calculate_campaign_metrics,
    get_campaign_feature_importance,
    prepare_campaign_model_data,
    train_and_evaluate_campaign_model,
)


def make_sample_data():
    return pd.DataFrame(
        {
            "income": [
                30000,
                40000,
                50000,
                60000,
                70000,
                80000,
                90000,
                100000,
            ],
            "recency": [
                80,
                70,
                60,
                50,
                40,
                30,
                20,
                10,
            ],
            "total_spend": [
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
            ],
            "total_purchases": [
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
            ],
            "total_children": [
                2,
                1,
                2,
                1,
                0,
                1,
                0,
                0,
            ],
            "household_size": [
                4,
                3,
                4,
                3,
                2,
                3,
                2,
                2,
            ],
            "num_deals_purchases": [
                2,
                2,
                1,
                1,
                1,
                0,
                0,
                0,
            ],
            "num_web_purchases": [
                2,
                3,
                3,
                4,
                5,
                6,
                7,
                8,
            ],
            "num_catalog_purchases": [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
            ],
            "num_store_purchases": [
                2,
                3,
                3,
                4,
                4,
                5,
                5,
                6,
            ],
            "num_web_visits_month": [
                8,
                8,
                7,
                7,
                6,
                6,
                5,
                5,
            ],
            "total_campaign_acceptances": [
                0,
                0,
                0,
                1,
                1,
                1,
                2,
                2,
            ],
            "web_purchase_share": [
                0.40,
                0.43,
                0.33,
                0.40,
                0.42,
                0.43,
                0.44,
                0.44,
            ],
            "catalog_purchase_share": [
                0.20,
                0.14,
                0.22,
                0.20,
                0.25,
                0.21,
                0.25,
                0.22,
            ],
            "store_purchase_share": [
                0.40,
                0.43,
                0.45,
                0.40,
                0.33,
                0.36,
                0.31,
                0.33,
            ],
            "response": [
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                0,
            ],
        }
    )


def test_default_campaign_features():
    assert len(
        DEFAULT_CAMPAIGN_FEATURES
    ) == 15

    assert "response" not in (
        DEFAULT_CAMPAIGN_FEATURES
    )

    assert "total_spend" in (
        DEFAULT_CAMPAIGN_FEATURES
    )


def test_prepare_campaign_model_data():
    df = make_sample_data()

    X, y = prepare_campaign_model_data(df)

    assert X.shape == (8, 15)
    assert y.shape == (8,)


def test_missing_income_is_preserved_for_pipeline_imputation():
    df = make_sample_data()

    df.loc[0, "income"] = np.nan

    X, y = prepare_campaign_model_data(df)

    # Imputation should happen inside the model pipeline,
    # not inside prepare_campaign_model_data().
    assert X["income"].isnull().sum() == 1
    assert len(X) == 8
    assert len(y) == 8


def test_missing_required_column_is_rejected():
    df = make_sample_data()

    df = df.drop(
        columns=["total_spend"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        prepare_campaign_model_data(df)


def test_invalid_target_is_rejected():
    df = make_sample_data()

    df.loc[0, "response"] = 2

    with pytest.raises(
        ValueError,
        match="Target must contain only 0 and 1",
    ):
        prepare_campaign_model_data(df)


def test_logistic_regression_can_train():
    df = make_sample_data()

    X, y = prepare_campaign_model_data(df)

    model = build_campaign_logistic_regression()

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_random_forest_can_train():
    df = make_sample_data()

    X, y = prepare_campaign_model_data(df)

    model = build_campaign_random_forest()

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_campaign_metrics():
    y_true = pd.Series(
        [0, 0, 1, 1]
    )

    y_pred = np.array(
        [0, 1, 1, 1]
    )

    y_probability = np.array(
        [0.1, 0.7, 0.8, 0.9]
    )

    metrics = calculate_campaign_metrics(
        y_true,
        y_pred,
        y_probability,
    )

    assert set(metrics.keys()) == {
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
    }

    for value in metrics.values():
        assert 0 <= value <= 1


def test_train_and_evaluate_campaign_model():
    df = make_sample_data()

    X, y = prepare_campaign_model_data(df)

    X_train = X.iloc[:6]
    y_train = y.iloc[:6]

    X_test = X.iloc[6:]
    y_test = y.iloc[6:]

    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}

    model = build_campaign_random_forest()

    (
        trained_model,
        metrics,
        matrix,
    ) = train_and_evaluate_campaign_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert trained_model is not None
    assert len(metrics) == 5
    assert matrix.shape == (2, 2)


def test_logistic_feature_importance():
    df = make_sample_data()

    X, y = prepare_campaign_model_data(df)

    model = build_campaign_logistic_regression()

    model.fit(X, y)

    importance = get_campaign_feature_importance(
        model,
        DEFAULT_CAMPAIGN_FEATURES,
    )

    assert len(importance) == 15

    assert list(
        importance.columns
    ) == [
        "feature",
        "importance",
    ]

    assert (
        importance["importance"]
        .notna()
        .all()
    )

    assert (
        importance["importance"] >= 0
    ).all()


def test_random_forest_feature_importance():
    df = make_sample_data()

    X, y = prepare_campaign_model_data(df)

    model = build_campaign_random_forest()

    model.fit(X, y)

    importance = get_campaign_feature_importance(
        model,
        DEFAULT_CAMPAIGN_FEATURES,
    )

    assert len(importance) == 15

    assert list(
        importance.columns
    ) == [
        "feature",
        "importance",
    ]

    assert (
        importance["importance"]
        .notna()
        .all()
    )

    assert (
        importance["importance"] >= 0
    ).all()