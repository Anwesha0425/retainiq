import pytest
from src.api.main import (
    health_check,
    root,
    predict_churn,
    predict_campaign_response,
    segment_rfm,
    get_customer_lifetime_value,
    simulate_campaign_roi,
    ChurnPredictionRequest,
    CampaignPredictionRequest,
    RFMSegmentRequest,
    CLVRequest,
    CampaignROISimulationRequest,
)


def test_root_endpoint():
    data = root()
    assert "message" in data
    assert "endpoints" in data


def test_health_check():
    data = health_check()
    assert data.status == "healthy"
    assert data.version == "1.0.0"


def test_predict_churn_high_risk():
    req = ChurnPredictionRequest(
        recency_days=180,
        frequency=1,
        monetary_total=45.0,
        active_months=1,
        units_total=5,
        unique_products=2,
    )
    res = predict_churn(req)
    assert res.churn_probability > 0.50
    assert res.is_churn_risk is True
    assert len(res.recommendations) > 0


def test_predict_churn_low_risk():
    req = ChurnPredictionRequest(
        recency_days=10,
        frequency=15,
        monetary_total=3500.0,
        active_months=12,
        units_total=100,
        unique_products=40,
    )
    res = predict_churn(req)
    assert res.is_churn_risk is False
    assert res.risk_tier == "Low Risk / Active"


def test_predict_campaign_response():
    req = CampaignPredictionRequest(
        recency_days=15,
        total_spend=1200.0,
        income=75000.0,
        web_purchases=6,
        catalog_purchases=5,
        accepted_previous_campaigns=2,
    )
    res = predict_campaign_response(req)
    assert res.response_probability > 0.20
    assert res.is_targeted is True
    assert res.targeting_priority in ["High", "Medium"]


def test_segment_rfm():
    req = RFMSegmentRequest(
        recency=10,
        frequency=15,
        monetary=4000.0,
    )
    res = segment_rfm(req)
    assert res.r_score == 5
    assert res.f_score == 5
    assert res.m_score == 5
    assert res.segment == "Champions"


def test_calculate_clv_endpoint():
    req = CLVRequest(
        frequency=6,
        monetary_value=1200.0,
        churn_rate=0.20,
        discount_rate=0.10,
    )
    res = get_customer_lifetime_value(req)
    assert res["estimated_clv"] > 0
    assert res["average_order_value"] == 200.0


def test_simulate_campaign_roi_endpoint():
    req = CampaignROISimulationRequest(
        total_customers=5000,
        selection_fraction=0.20,
        base_response_rate=0.15,
        model_lift_multiplier=2.0,
        revenue_per_conversion=50.0,
        cost_per_outreach=1.5,
    )
    res = simulate_campaign_roi(req)
    assert res["profit_lift"] > 0
    assert res["targeted_customers"] == 1000
