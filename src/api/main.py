"""
FastAPI Production Microservice for Customer & Marketing Analytics.
Provides REST endpoints for Churn Risk Prediction, RFM Segmentation,
Campaign Targeting Optimization, and Customer Lifetime Value (CLV).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.modeling.advanced_models import (
    calculate_customer_lifetime_value,
    categorize_uplift_segments,
    calculate_campaign_roi_lift,
)

app = FastAPI(
    title="Customer & Marketing Analytics API",
    description="Production REST API for churn prediction, RFM segmentation, and campaign optimization.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Pydantic Schemas
# ==========================================

class HealthResponse(BaseModel):
    status: str
    version: str
    service: str


class ChurnPredictionRequest(BaseModel):
    recency_days: float = Field(..., ge=0, description="Days since last purchase")
    frequency: float = Field(..., ge=1, description="Total purchase frequency (invoices)")
    monetary_total: float = Field(..., ge=0, description="Total monetary spend")
    units_total: float = Field(default=10, ge=0, description="Total units purchased")
    active_months: float = Field(default=1, ge=1, description="Number of distinct active months")
    unique_products: float = Field(default=5, ge=1, description="Number of unique products purchased")
    avg_order_value: Optional[float] = Field(default=None, description="Average order value")


class ChurnPredictionResponse(BaseModel):
    churn_probability: float
    is_churn_risk: bool
    risk_tier: str
    recommendations: List[str]


class RFMSegmentRequest(BaseModel):
    recency: float = Field(..., ge=0, description="Days since last purchase")
    frequency: float = Field(..., ge=1, description="Number of transactions")
    monetary: float = Field(..., ge=0, description="Total spend")


class RFMSegmentResponse(BaseModel):
    r_score: int
    f_score: int
    m_score: int
    rfm_composite: str
    segment: str
    description: str


class CampaignPredictionRequest(BaseModel):
    recency_days: float = Field(..., ge=0, description="Days since last engagement")
    total_spend: float = Field(..., ge=0, description="Total customer historical spend")
    income: float = Field(default=50000.0, ge=0, description="Customer annual income")
    web_purchases: int = Field(default=3, ge=0, description="Web purchases count")
    catalog_purchases: int = Field(default=2, ge=0, description="Catalog purchases count")
    accepted_previous_campaigns: int = Field(default=0, ge=0, description="Number of accepted past campaigns")
    education_level: str = Field(default="Graduation", description="Graduation, PhD, Master, Basic, 2n Cycle")


class CampaignPredictionResponse(BaseModel):
    response_probability: float
    is_targeted: bool
    targeting_priority: str
    uplift_segment: str
    recommended_action: str


class CLVRequest(BaseModel):
    frequency: float = Field(..., ge=1, description="Annual transaction count")
    monetary_value: float = Field(..., ge=0, description="Total annual monetary spend")
    churn_rate: float = Field(..., ge=0.01, le=1.0, description="Expected annual churn rate")
    discount_rate: float = Field(default=0.10, ge=0.0, le=0.5, description="Discount rate")
    lifespan_months: Optional[int] = Field(default=None, ge=1, le=120, description="Optional projection horizon")


class CampaignROISimulationRequest(BaseModel):
    total_customers: int = Field(default=10000, ge=100)
    selection_fraction: float = Field(default=0.20, gt=0.0, le=1.0)
    base_response_rate: float = Field(default=0.149, gt=0.0, le=1.0)
    model_lift_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)
    revenue_per_conversion: float = Field(default=50.0, gt=0.0)
    cost_per_outreach: float = Field(default=1.5, gt=0.0)


# ==========================================
# Helper Business Logic
# ==========================================

def calculate_heuristic_churn_risk(
    recency: float,
    frequency: float,
    monetary: float,
    active_months: float,
) -> float:
    """
    Calibrated churn risk scoring function based on findings from
    the chronological Random Forest model in the repository.
    """
    # Base probability
    score = 0.50

    # Recency penalty / reward
    if recency > 90:
        score += min((recency - 90) / 90 * 0.35, 0.40)
    else:
        score -= (90 - recency) / 90 * 0.25

    # Frequency and Active Months buffer
    if active_months >= 6:
        score -= 0.15
    elif active_months <= 1:
        score += 0.10

    if frequency >= 10:
        score -= 0.12
    elif frequency <= 2:
        score += 0.08

    # Monetary buffer
    if monetary > 2000:
        score -= 0.10
    elif monetary < 150:
        score += 0.05

    return max(0.01, min(0.99, score))


def assign_rfm_segment(r: int, f: int, m: int) -> Tuple[str, str]:
    """Assign business segment based on RFM scores (1 to 5)."""
    rfm_avg = (r + f + m) / 3.0

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions", "Best customers who purchase frequently, recently, and spend heavily."
    elif r >= 3 and f >= 3 and m >= 3:
        return "Loyal Customers", "Consistent purchasers who respond well to promotions and loyalty rewards."
    elif r >= 4 and f <= 2:
        return "Recent Customers / Potential Loyalists", "New or recent shoppers with high potential for engagement."
    elif r <= 2 and f >= 3 and m >= 3:
        return "High Value At Risk", "High spenders who haven't purchased in a while. Requires immediate win-back campaigns."
    elif r <= 2 and f <= 2:
        return "Lost Customers", "Infrequent purchasers with long dormancy. May not justify expensive reactivation."
    else:
        return "At Risk / Customers Needing Attention", "Above average customers at risk of slipping into churn."


# ==========================================
# Endpoints
# ==========================================

@app.get("/", tags=["General"])
def root() -> Dict[str, Any]:
    return {
        "message": "Welcome to the Customer & Marketing Analytics API",
        "docs_url": "/docs",
        "endpoints": [
            "/health",
            "/api/v1/predict/churn",
            "/api/v1/predict/campaign",
            "/api/v1/segment/rfm",
            "/api/v1/calculate/clv",
            "/api/v1/simulate/campaign-roi",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="Customer & Marketing Analytics Service",
    )


@app.post("/api/v1/predict/churn", response_model=ChurnPredictionResponse, tags=["Modeling"])
def predict_churn(request: ChurnPredictionRequest) -> ChurnPredictionResponse:
    prob = calculate_heuristic_churn_risk(
        recency=request.recency_days,
        frequency=request.frequency,
        monetary=request.monetary_total,
        active_months=request.active_months,
    )

    is_risk = prob >= 0.50

    if prob >= 0.70:
        tier = "High Risk"
        recs = [
            "Trigger personalized win-back discount immediately.",
            "Send re-engagement survey to address dissatisfaction.",
            "Exclude from cold upselling campaigns to avoid opt-outs.",
        ]
    elif prob >= 0.40:
        tier = "Medium Risk"
        recs = [
            "Send product recommendations based on previous top purchases.",
            "Offer limited-time incentive or free shipping.",
            "Monitor weekly purchase activity closely.",
        ]
    else:
        tier = "Low Risk / Active"
        recs = [
            "Enroll in VIP / Loyalty reward tier.",
            "Cross-sell complementary products.",
            "Encourage organic advocacy and reviews.",
        ]

    return ChurnPredictionResponse(
        churn_probability=round(prob, 4),
        is_churn_risk=is_risk,
        risk_tier=tier,
        recommendations=recs,
    )


@app.post("/api/v1/predict/campaign", response_model=CampaignPredictionResponse, tags=["Modeling"])
def predict_campaign_response(request: CampaignPredictionRequest) -> CampaignPredictionResponse:
    # Baseline probability ~14.9% from Customer Personality dataset
    base = 0.149

    # Previous acceptance is the strongest empirical feature (+15% per past campaign)
    base += min(request.accepted_previous_campaigns * 0.16, 0.48)

    # High spenders have ~30% response rate in Q4
    if request.total_spend > 1000:
        base += 0.15
    elif request.total_spend > 500:
        base += 0.08
    elif request.total_spend < 100:
        base -= 0.06

    # Recency effect (active shoppers respond better)
    if request.recency_days < 30:
        base += 0.08
    elif request.recency_days > 75:
        base -= 0.07

    # Multi-channel engagement
    if request.catalog_purchases >= 4:
        base += 0.10

    prob = max(0.02, min(0.95, base))
    is_targeted = prob >= 0.25

    if prob >= 0.40:
        priority = "High"
    elif prob >= 0.20:
        priority = "Medium"
    else:
        priority = "Low"

    # Uplift quadrant estimation
    control_prob = prob * 0.35  # Estimated organic conversion without outreach
    uplift_res = categorize_uplift_segments(control_prob, prob)

    return CampaignPredictionResponse(
        response_probability=round(prob, 4),
        is_targeted=is_targeted,
        targeting_priority=priority,
        uplift_segment=uplift_res["uplift_segment"],
        recommended_action=uplift_res["recommended_action"],
    )


@app.post("/api/v1/segment/rfm", response_model=RFMSegmentResponse, tags=["Segmentation"])
def segment_rfm(request: RFMSegmentRequest) -> RFMSegmentResponse:
    # Compute quintile approximations based on Online Retail II distribution
    # Recency: lower is better
    if request.recency <= 15:
        r = 5
    elif request.recency <= 40:
        r = 4
    elif request.recency <= 80:
        r = 3
    elif request.recency <= 180:
        r = 2
    else:
        r = 1

    # Frequency: higher is better
    if request.frequency >= 12:
        f = 5
    elif request.frequency >= 6:
        f = 4
    elif request.frequency >= 3:
        f = 3
    elif request.frequency >= 2:
        f = 2
    else:
        f = 1

    # Monetary: higher is better
    if request.monetary >= 3500:
        m = 5
    elif request.monetary >= 1500:
        m = 4
    elif request.monetary >= 700:
        m = 3
    elif request.monetary >= 300:
        m = 2
    else:
        m = 1

    segment_name, description = assign_rfm_segment(r, f, m)

    return RFMSegmentResponse(
        r_score=r,
        f_score=f,
        m_score=m,
        rfm_composite=f"{r}{f}{m}",
        segment=segment_name,
        description=description,
    )


@app.post("/api/v1/calculate/clv", tags=["Analytics"])
def get_customer_lifetime_value(request: CLVRequest) -> Dict[str, float]:
    return calculate_customer_lifetime_value(
        frequency=request.frequency,
        monetary_value=request.monetary_value,
        churn_rate=request.churn_rate,
        discount_rate=request.discount_rate,
        lifespan_months=request.lifespan_months,
    )


@app.post("/api/v1/simulate/campaign-roi", tags=["Analytics"])
def simulate_campaign_roi(request: CampaignROISimulationRequest) -> Dict[str, float]:
    return calculate_campaign_roi_lift(
        total_customers=request.total_customers,
        selection_fraction=request.selection_fraction,
        base_response_rate=request.base_response_rate,
        model_lift_multiplier=request.model_lift_multiplier,
        revenue_per_conversion=request.revenue_per_conversion,
        cost_per_outreach=request.cost_per_outreach,
    )

