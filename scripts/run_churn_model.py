from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sklearn.model_selection import train_test_split

from src.modeling.churn_model import (
    DEFAULT_FEATURES,
    build_logistic_regression_model,
    build_random_forest_model,
    get_feature_importance,
    prepare_model_data,
    train_and_evaluate_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

load_dotenv(PROJECT_ROOT / ".env")


def get_database_engine():
    """Create PostgreSQL SQLAlchemy engine."""

    import os

    database_url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
    )

    return create_engine(database_url)


def load_time_based_churn_data(engine) -> pd.DataFrame:
    """Load the leakage-safe time-based churn dataset."""

    query = text(
        """
        SELECT *
        FROM time_based_churn_model
        ORDER BY customer_id
        """
    )

    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Create a stratified holdout split.

    The features themselves are already constructed using only
    information available before the prediction cutoff.
    """

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


def save_metrics(
    metrics: dict,
    model_name: str,
):
    """Save model evaluation metrics."""

    output = pd.DataFrame(
        [
            {
                "model": model_name,
                **metrics,
            }
        ]
    )

    output_path = RESULTS_DIR / "churn_model_metrics.csv"

    if output_path.exists():
        existing = pd.read_csv(output_path)
        existing = existing[
            existing["model"] != model_name
        ]
        output = pd.concat(
            [existing, output],
            ignore_index=True,
        )

    output.to_csv(output_path, index=False)


def save_confusion_matrix(
    matrix,
    model_name: str,
):
    """Save confusion matrix."""

    output = pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )

    output["model"] = model_name

    output_path = (
        RESULTS_DIR
        / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.csv"
    )

    output.to_csv(output_path)


def save_feature_importance(
    importance: pd.DataFrame,
    model_name: str,
):
    """Save feature importance."""

    output = importance.copy()
    output["model"] = model_name

    output_path = (
        RESULTS_DIR
        / f"{model_name.lower().replace(' ', '_')}_feature_importance.csv"
    )

    output.to_csv(output_path, index=False)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading time-based churn dataset...")

    engine = get_database_engine()

    try:
        df = load_time_based_churn_data(engine)
    finally:
        engine.dispose()

    print(f"Rows loaded: {len(df):,}")

    X, y = prepare_model_data(
        df,
        feature_columns=DEFAULT_FEATURES,
        target_column="churn",
    )

    print("\nFeature columns:")
    for feature in DEFAULT_FEATURES:
        print(f"  - {feature}")

    print("\nChurn distribution:")
    print(y.value_counts().sort_index())

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(X, y)

    print("\nTrain/Test split:")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows : {len(X_test):,}")

    models = {
        "Logistic Regression": build_logistic_regression_model(),
        "Random Forest": build_random_forest_model(),
    }

    for model_name, model in models.items():

        print(f"\n{'=' * 60}")
        print(model_name)
        print(f"{'=' * 60}")

        (
            trained_model,
            metrics,
            confusion,
        ) = train_and_evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        print("\nMetrics:")

        for metric, value in metrics.items():
            print(f"{metric:12s}: {value:.4f}")

        print("\nConfusion Matrix:")
        print(confusion)

        save_metrics(
            metrics,
            model_name,
        )

        save_confusion_matrix(
            confusion,
            model_name,
        )

        importance = get_feature_importance(
            trained_model,
            DEFAULT_FEATURES,
        )

        save_feature_importance(
            importance,
            model_name,
        )

        print("\nTop features:")

        print(
            importance.head(10).to_string(
                index=False
            )
        )

    print("\nMODEL TRAINING COMPLETED")


if __name__ == "__main__":
    main()