from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv
import os

from src.modeling.campaign_model import (
    DEFAULT_CAMPAIGN_FEATURES,
    build_campaign_logistic_regression,
    build_campaign_random_forest,
    get_campaign_feature_importance,
    prepare_campaign_model_data,
    train_and_evaluate_campaign_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

load_dotenv(PROJECT_ROOT / ".env")


def get_database_engine():
    """Create a PostgreSQL SQLAlchemy engine."""

    database_url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
    )

    return create_engine(database_url)


def load_campaign_data(engine) -> pd.DataFrame:
    """Load marketing customer features from PostgreSQL."""

    query = """
        SELECT *
        FROM marketing_customer_features
    """

    return pd.read_sql(
        query,
        engine,
    )


def save_model_results(
    model_name: str,
    metrics: dict[str, float],
    feature_importance: pd.DataFrame,
    confusion: object,
):
    """Save metrics, feature importance and confusion matrix."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "model": model_name,
                **metrics,
            }
        ]
    )

    metrics_file = (
        RESULTS_DIR
        / "campaign_model_metrics.csv"
    )

    if metrics_file.exists():
        existing_metrics = pd.read_csv(
            metrics_file
        )

        existing_metrics = existing_metrics[
            existing_metrics["model"]
            != model_name
        ]

        metrics_df = pd.concat(
            [
                existing_metrics,
                metrics_df,
            ],
            ignore_index=True,
        )

    metrics_df.to_csv(
        metrics_file,
        index=False,
    )

    safe_name = model_name.lower().replace(
        " ",
        "_",
    )

    feature_importance.to_csv(
        RESULTS_DIR
        / f"campaign_{safe_name}_feature_importance.csv",
        index=False,
    )

    confusion_df = pd.DataFrame(
        confusion,
        index=[
            "actual_0",
            "actual_1",
        ],
        columns=[
            "predicted_0",
            "predicted_1",
        ],
    )

    confusion_df.to_csv(
        RESULTS_DIR
        / f"campaign_{safe_name}_confusion_matrix.csv"
    )


def main():
    print("=" * 60)
    print("Campaign Response Modeling")
    print("=" * 60)

    engine = get_database_engine()

    print("\nLoading campaign data...")

    try:
        df = load_campaign_data(engine)
    finally:
        engine.dispose()

    print(
        f"Loaded {len(df):,} customers."
    )

    X, y = prepare_campaign_model_data(
        df,
        DEFAULT_CAMPAIGN_FEATURES,
        "response",
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        f"Response distribution:\n{y.value_counts().sort_index()}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(
        f"\nTraining rows: {len(X_train):,}"
    )

    print(
        f"Testing rows: {len(X_test):,}"
    )

    models = {
        "Logistic Regression":
            build_campaign_logistic_regression(),

        "Random Forest":
            build_campaign_random_forest(),
    }

    for model_name, model in models.items():

        print("\n" + "-" * 60)
        print(model_name)
        print("-" * 60)

        (
            trained_model,
            metrics,
            confusion,
        ) = train_and_evaluate_campaign_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        print("\nMetrics:")

        for metric, value in metrics.items():

            print(
                f"{metric.upper():<12}: "
                f"{value:.4f}"
            )

        print("\nConfusion Matrix:")

        print(confusion)

        feature_importance = (
            get_campaign_feature_importance(
                trained_model,
                DEFAULT_CAMPAIGN_FEATURES,
            )
        )

        print("\nTop 10 Features:")

        print(
            feature_importance.head(10).to_string(
                index=False
            )
        )

        save_model_results(
            model_name,
            metrics,
            feature_importance,
            confusion,
        )

    print("\n" + "=" * 60)
    print("CAMPAIGN MODELING COMPLETED")
    print("=" * 60)

    print("\nResults saved to:")

    print(
        RESULTS_DIR
    )


if __name__ == "__main__":
    main()