import pytest
import pandas as pd
import numpy as np

from src.modeling.advanced_models import (
    calculate_customer_lifetime_value,
    categorize_uplift_segments,
    calculate_campaign_roi_lift,
    train_xgboost_classifier,
)


def test_calculate_customer_lifetime_value():
    clv_result = calculate_customer_lifetime_value(
        frequency=5.0,
        monetary_value=1000.0,
        churn_rate=0.20,
        discount_rate=0.10,
    )

    assert "estimated_clv" in clv_result
    assert "average_order_value" in clv_result
    assert clv_result["average_order_value"] == 200.0
    assert clv_result["estimated_clv"] > 0


def test_calculate_customer_lifetime_value_with_horizon():
    clv_finite = calculate_customer_lifetime_value(
        frequency=4.0,
        monetary_value=800.0,
        churn_rate=0.25,
        discount_rate=0.10,
        lifespan_months=24,
    )

    assert clv_finite["estimated_clv"] > 0
    assert clv_finite["average_order_value"] == 200.0


def test_categorize_uplift_segments():
    # Persuadable case
    persuadable = categorize_uplift_segments(
        control_response_prob=0.05,
        treatment_response_prob=0.35,
    )
    assert persuadable["uplift_segment"] == "Persuadables"

    # Sure Thing case
    sure_thing = categorize_uplift_segments(
        control_response_prob=0.60,
        treatment_response_prob=0.65,
        threshold=0.15,
    )
    assert sure_thing["uplift_segment"] == "Sure Things"

    # Lost Cause case
    lost_cause = categorize_uplift_segments(
        control_response_prob=0.05,
        treatment_response_prob=0.10,
        threshold=0.15,
    )
    assert lost_cause["uplift_segment"] == "Lost Causes"


def test_calculate_campaign_roi_lift():
    roi_result = calculate_campaign_roi_lift(
        total_customers=1000,
        selection_fraction=0.20,
        base_response_rate=0.15,
        model_lift_multiplier=2.0,
        revenue_per_conversion=50.0,
        cost_per_outreach=2.0,
    )

    assert roi_result["targeted_customers"] == 200
    assert roi_result["profit_lift"] > 0
    assert roi_result["targeted_roi_pct"] > roi_result["random_roi_pct"]


def test_train_xgboost_classifier():
    np.random.seed(42)
    x = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
    })
    y = pd.Series((x["feature1"] + x["feature2"] > 0).astype(int))

    x_train, x_test = x.iloc[:80], x.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    result = train_xgboost_classifier(
        x_train, y_train, x_test, y_test, n_estimators=10, max_depth=2
    )

    assert "model" in result
    assert "metrics" in result
    assert "feature_importances" in result
    assert result["metrics"]["roc_auc"] >= 0.0

