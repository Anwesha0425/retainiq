import numpy as np
import pandas as pd
import pytest

from src.modeling.churn_model import (
    DEFAULT_FEATURES,
    build_logistic_regression_model,
    build_random_forest_model,
    calculate_classification_metrics,
    get_feature_importance,
    prepare_model_data,
    train_and_evaluate_model,
)


def make_sample_data():
    return pd.DataFrame(
        {
            "purchase_frequency": [10, 8, 2, 1, 7, 3, 9, 2],
            "total_spend": [1000, 800, 200, 100, 700, 300, 900, 150],
            "average_line_value": [20, 18, 15, 10, 19, 12, 21, 11],
            "active_purchase_months": [10, 8, 2, 1, 7, 3, 9, 2],
            "unique_products_purchased": [50, 40, 10, 5, 35, 12, 45, 8],
            "total_units_purchased": [500, 400, 80, 30, 350, 100, 450, 60],
            "total_purchase_lines": [100, 80, 20, 10, 70, 25, 90, 15],
            "return_lines": [2, 1, 0, 0, 1, 0, 2, 0],
            "returned_units": [3, 1, 0, 0, 2, 0, 3, 0],
            "average_order_value": [100, 100, 100, 100, 100, 100, 100, 100],
            "return_line_rate": [0.02, 0.01, 0, 0, 0.01, 0, 0.02, 0],
            "churn": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )


def test_default_features_are_defined():
    assert len(DEFAULT_FEATURES) == 11
    assert "purchase_frequency" in DEFAULT_FEATURES
    assert "total_spend" in DEFAULT_FEATURES
    assert "churn" not in DEFAULT_FEATURES


def test_prepare_model_data_returns_correct_shapes():
    df = make_sample_data()

    X, y = prepare_model_data(df)

    assert X.shape == (8, 11)
    assert y.shape == (8,)


def test_prepare_model_data_rejects_missing_columns():
    df = make_sample_data().drop(columns=["total_spend"])

    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_model_data(df)


def test_prepare_model_data_rejects_missing_values():
    df = make_sample_data()
    df.loc[0, "total_spend"] = np.nan

    with pytest.raises(
        ValueError,
        match="Feature matrix contains missing values",
    ):
        prepare_model_data(df)


def test_prepare_model_data_rejects_invalid_target():
    df = make_sample_data()
    df.loc[0, "churn"] = 2

    with pytest.raises(
        ValueError,
        match="Target must contain only 0 and 1",
    ):
        prepare_model_data(df)


def test_logistic_regression_model_can_train():
    df = make_sample_data()
    X, y = prepare_model_data(df)

    model = build_logistic_regression_model()

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_random_forest_model_can_train():
    df = make_sample_data()
    X, y = prepare_model_data(df)

    model = build_random_forest_model()

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_classification_metrics_are_generated():
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_probability = np.array([0.1, 0.7, 0.8, 0.9])

    metrics = calculate_classification_metrics(
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


def test_train_and_evaluate_model():
    df = make_sample_data()

    X, y = prepare_model_data(df)

    X_train = X.iloc[:6]
    y_train = y.iloc[:6]

    X_test = X.iloc[6:]
    y_test = y.iloc[6:]

    model = build_random_forest_model()

    trained_model, metrics, matrix = train_and_evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert trained_model is not None
    assert len(metrics) == 5
    assert matrix.shape == (2, 2)


def test_logistic_regression_feature_importance():
    df = make_sample_data()

    X, y = prepare_model_data(df)

    model = build_logistic_regression_model()
    model.fit(X, y)

    importance = get_feature_importance(
        model,
        DEFAULT_FEATURES,
    )

    assert len(importance) == 11
    assert list(importance.columns) == [
        "feature",
        "importance",
    ]


def test_random_forest_feature_importance():
    df = make_sample_data()

    X, y = prepare_model_data(df)

    model = build_random_forest_model()
    model.fit(X, y)

    importance = get_feature_importance(
        model,
        DEFAULT_FEATURES,
    )

    assert len(importance) == 11
    assert list(importance.columns) == [
        "feature",
        "importance",
    ]