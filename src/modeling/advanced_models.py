"""
Advanced modeling, explainability, customer lifetime value (CLV),
and uplift marketing optimization module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def train_xgboost_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train an XGBoost classifier with early stopping and return
    fitted model, test metrics, and feature importances.
    """
    if xgb is None:
        raise ImportError("xgboost is not installed. Please install xgboost.")

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric="logloss",
    )
    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)
    probs = clf.predict_proba(x_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "pr_auc": float(average_precision_score(y_test, probs)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }

    feature_importances = pd.DataFrame({
        "feature": x_train.columns,
        "importance": clf.feature_importances_,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return {
        "model": clf,
        "metrics": metrics,
        "feature_importances": feature_importances,
    }


def calculate_customer_lifetime_value(
    frequency: float,
    monetary_value: float,
    churn_rate: float,
    discount_rate: float = 0.10,
    lifespan_months: Optional[int] = None,
) -> Dict[str, float]:
    """
    Calculate estimated Customer Lifetime Value (CLV).
    
    Uses standard retention-based CLV model:
      CLV = (Average Order Value * Frequency * Profit Margin) / (Churn Rate + Discount Rate)
      or finite horizon discounted cash flow when lifespan_months is specified.
    """
    if churn_rate <= 0:
        churn_rate = 0.01  # Prevent division by zero
    if churn_rate > 1:
        churn_rate = 1.0

    avg_order_value = monetary_value / max(frequency, 1.0)
    profit_margin = 0.30  # Standard retail gross margin assumption

    if lifespan_months:
        # Finite horizon calculation
        monthly_churn = 1.0 - (1.0 - churn_rate) ** (1.0 / 12.0)
        monthly_discount = (1.0 + discount_rate) ** (1.0 / 12.0) - 1.0
        clv = 0.0
        retention = 1.0
        monthly_spend = (monetary_value * profit_margin) / 12.0
        for m in range(1, lifespan_months + 1):
            clv += (monthly_spend * retention) / ((1.0 + monthly_discount) ** m)
            retention *= (1.0 - monthly_churn)
    else:
        # Infinite horizon capitalized CLV
        clv = (monetary_value * profit_margin) / (churn_rate + discount_rate)

    return {
        "estimated_clv": round(float(clv), 2),
        "average_order_value": round(float(avg_order_value), 2),
        "expected_margin": round(float(profit_margin), 2),
        "annual_spend": round(float(monetary_value), 2),
    }


def categorize_uplift_segments(
    control_response_prob: float,
    treatment_response_prob: float,
    threshold: float = 0.15,
) -> Dict[str, Any]:
    """
    Categorize customer into uplift marketing quadrant:
    - Persuadables: Respond only when treated (treatment >> control)
    - Sure Things: Respond regardless of treatment (treatment ~ control and high)
    - Lost Causes: Do not respond in either state (treatment ~ control and low)
    - Sleeping Dogs: Respond negatively when treated (treatment << control)
    """
    uplift = treatment_response_prob - control_response_prob

    if uplift > threshold:
        segment = "Persuadables"
        action = "High Priority: Target with active campaign promotions."
    elif treatment_response_prob > 0.50 and control_response_prob > 0.50:
        segment = "Sure Things"
        action = "Organic Retention: Do not spend budget; they will buy organically."
    elif treatment_response_prob <= 0.20 and control_response_prob <= 0.20:
        segment = "Lost Causes"
        action = "Deprioritize: Unlikely to convert regardless of outreach."
    else:
        segment = "Sleeping Dogs"
        action = "Caution: Avoid intrusive outreach; may increase unsubscribe rate."

    return {
        "uplift_score": round(float(uplift), 4),
        "uplift_segment": segment,
        "recommended_action": action,
    }


def calculate_campaign_roi_lift(
    total_customers: int,
    selection_fraction: float,
    base_response_rate: float,
    model_lift_multiplier: float,
    revenue_per_conversion: float = 50.0,
    cost_per_outreach: float = 1.5,
) -> Dict[str, float]:
    """
    Simulate expected campaign profit and ROI lift using ML targeting vs random selection.
    """
    targeted_customers = int(total_customers * selection_fraction)
    random_response_rate = base_response_rate
    targeted_response_rate = min(base_response_rate * model_lift_multiplier, 1.0)

    # ML targeting outcome
    targeted_conversions = targeted_customers * targeted_response_rate
    targeted_revenue = targeted_conversions * revenue_per_conversion
    targeted_cost = targeted_customers * cost_per_outreach
    targeted_profit = targeted_revenue - targeted_cost
    targeted_roi = (targeted_profit / targeted_cost) * 100 if targeted_cost > 0 else 0.0

    # Random targeting outcome (same budget)
    random_conversions = targeted_customers * random_response_rate
    random_revenue = random_conversions * revenue_per_conversion
    random_cost = targeted_cost
    random_profit = random_revenue - random_cost
    random_roi = (random_profit / random_cost) * 100 if random_cost > 0 else 0.0

    profit_lift = targeted_profit - random_profit

    return {
        "targeted_customers": targeted_customers,
        "targeted_conversions": round(targeted_conversions, 1),
        "targeted_profit": round(targeted_profit, 2),
        "targeted_roi_pct": round(targeted_roi, 1),
        "random_conversions": round(random_conversions, 1),
        "random_profit": round(random_profit, 2),
        "random_roi_pct": round(random_roi, 1),
        "profit_lift": round(profit_lift, 2),
    }
