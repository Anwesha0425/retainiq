from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "purchase_frequency",
    "total_spend",
    "average_line_value",
    "active_purchase_months",
    "unique_products_purchased",
    "total_units_purchased",
    "total_purchase_lines",
    "return_lines",
    "returned_units",
    "average_order_value",
    "return_line_rate",
]


def prepare_model_data(
    df: pd.DataFrame,
    feature_columns: Iterable[str] = DEFAULT_FEATURES,
    target_column: str = "churn",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target vector y.

    The input dataframe is expected to come from the
    leakage-safe time_based_churn_model dataset.
    """

    feature_columns = list(feature_columns)

    required_columns = set(feature_columns) | {target_column}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    if X.isnull().any().any():
        raise ValueError("Feature matrix contains missing values.")

    if y.isnull().any():
        raise ValueError("Target contains missing values.")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    return X, y


def build_logistic_regression_model() -> Pipeline:
    """
    Build a standardized Logistic Regression model.

    Logistic Regression acts as an interpretable baseline.
    """

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest_model() -> RandomForestClassifier:
    """
    Build a Random Forest classifier.
    """

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def calculate_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    """
    Calculate classification metrics for a binary churn model.
    """

    return {
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(average_precision_score(y_true, y_probability)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
    }


def train_and_evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[object, dict[str, float], np.ndarray]:
    """
    Train a model and evaluate it on the test set.

    Returns:
        trained model
        classification metrics
        confusion matrix
    """

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = calculate_classification_metrics(
        y_test,
        y_pred,
        y_probability,
    )

    matrix = confusion_matrix(y_test, y_pred)

    return model, metrics, matrix


def get_feature_importance(
    model,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    Supports:
    - Logistic Regression Pipeline
    - Random Forest
    """

    feature_columns = list(feature_columns)

    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]

        if not hasattr(estimator, "coef_"):
            raise ValueError(
                "Pipeline model does not expose coefficients."
            )

        importance = np.abs(estimator.coef_[0])

    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    else:
        raise ValueError(
            "Model does not expose feature importance."
        )

    result = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance,
        }
    )

    return result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)