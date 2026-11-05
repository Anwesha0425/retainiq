from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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


DEFAULT_CAMPAIGN_FEATURES = [
    "income",
    "recency",
    "total_spend",
    "total_purchases",
    "total_children",
    "household_size",
    "num_deals_purchases",
    "num_web_purchases",
    "num_catalog_purchases",
    "num_store_purchases",
    "num_web_visits_month",
    "total_campaign_acceptances",
    "web_purchase_share",
    "catalog_purchase_share",
    "store_purchase_share",
]


def prepare_campaign_model_data(
    df: pd.DataFrame,
    feature_columns: Iterable[str] = DEFAULT_CAMPAIGN_FEATURES,
    target_column: str = "response",
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare campaign-response model features and target.

    Missing feature values are intentionally left as NaN.
    Imputation is performed inside the model pipeline so that
    imputation statistics are learned only from the training data.
    """

    feature_columns = list(feature_columns)

    required_columns = set(feature_columns) | {target_column}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data = df[feature_columns + [target_column]].copy()

    # All model features must be numeric.
    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(
                f"Campaign feature must be numeric: {column}"
            )

    X = data[feature_columns].copy()
    y = data[target_column].copy()

    if y.isnull().any():
        raise ValueError("Target contains missing values.")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "Target must contain only 0 and 1."
        )

    return X, y


def build_campaign_logistic_regression() -> Pipeline:
    """Build an interpretable Logistic Regression baseline.

    Median imputation is fitted only on training data.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
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


def build_campaign_random_forest() -> Pipeline:
    """Build a Random Forest campaign-response model.

    Median imputation is fitted only on training data.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def calculate_campaign_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    """Calculate campaign-response classification metrics."""

    if y_true.nunique() < 2:
        raise ValueError(
            "ROC-AUC requires both classes to be present in y_true."
        )

    return {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                y_probability,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
    }


def train_and_evaluate_campaign_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, dict[str, float], np.ndarray]:
    """Train and evaluate a campaign-response model."""

    if y_train.nunique() < 2:
        raise ValueError(
            "Training target must contain both classes."
        )

    if y_test.nunique() < 2:
        raise ValueError(
            "Test target must contain both classes."
        )

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    metrics = calculate_campaign_metrics(
        y_test,
        y_pred,
        y_probability,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    return (
        model,
        metrics,
        matrix,
    )


def get_campaign_feature_importance(
    model,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """Extract feature importance from a trained campaign model."""

    feature_columns = list(feature_columns)

    if isinstance(model, Pipeline):

        estimator = model.named_steps["model"]

        if hasattr(estimator, "coef_"):

            importance = np.abs(
                estimator.coef_[0]
            )

        elif hasattr(
            estimator,
            "feature_importances_",
        ):

            importance = estimator.feature_importances_

        else:
            raise ValueError(
                "Pipeline model does not expose feature importance."
            )

    elif hasattr(
        model,
        "feature_importances_",
    ):

        importance = model.feature_importances_

    else:

        raise ValueError(
            "Model does not expose feature importance."
        )

    if len(importance) != len(feature_columns):
        raise ValueError(
            "Number of feature importance values does not "
            "match number of feature columns."
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