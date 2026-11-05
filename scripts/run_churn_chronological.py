from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

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


def load_snapshot_data(engine) -> pd.DataFrame:
    """Load chronological churn snapshots from PostgreSQL."""

    query = text(
        """
        SELECT *
        FROM churn_model_snapshots
        ORDER BY cutoff_date, customer_id
        """
    )

    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def split_by_cutoff(
    df: pd.DataFrame,
    test_cutoff: str = "2011-09-01",
):
    """
    Split data chronologically.

    All snapshots before test_cutoff are training data.
    test_cutoff itself is the final holdout period.
    """

    df = df.copy()

    # PostgreSQL DATE values may be returned as datetime.date.
    # Normalize them to pandas datetime64 for reliable comparisons.
    df["cutoff_date"] = pd.to_datetime(
        df["cutoff_date"]
    )

    test_cutoff = pd.Timestamp(test_cutoff)

    train_df = df[
        df["cutoff_date"] < test_cutoff
    ].copy()

    test_df = df[
        df["cutoff_date"] == test_cutoff
    ].copy()

    if train_df.empty:
        raise ValueError("Training dataset is empty.")

    if test_df.empty:
        raise ValueError("Test dataset is empty.")

    return train_df, test_df


def save_metrics(
    metrics: dict,
    model_name: str,
):
    """Save chronological model metrics."""

    output = pd.DataFrame(
        [
            {
                "model": model_name,
                **metrics,
            }
        ]
    )

    output_path = (
        RESULTS_DIR
        / "chronological_churn_model_metrics.csv"
    )

    if output_path.exists():
        existing = pd.read_csv(output_path)

        existing = existing[
            existing["model"] != model_name
        ]

        output = pd.concat(
            [existing, output],
            ignore_index=True,
        )

    output.to_csv(
        output_path,
        index=False,
    )


def save_confusion_matrix(
    matrix,
    model_name: str,
):
    """Save chronological confusion matrix."""

    output = pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )

    output_path = (
        RESULTS_DIR
        / (
            model_name.lower().replace(" ", "_")
            + "_chronological_confusion_matrix.csv"
        )
    )

    output.to_csv(output_path)


def save_feature_importance(
    importance: pd.DataFrame,
    model_name: str,
):
    """Save chronological feature importance."""

    output = importance.copy()

    output["model"] = model_name

    output_path = (
        RESULTS_DIR
        / (
            model_name.lower().replace(" ", "_")
            + "_chronological_feature_importance.csv"
        )
    )

    output.to_csv(
        output_path,
        index=False,
    )


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading chronological churn snapshots...")

    engine = get_database_engine()

    try:
        df = load_snapshot_data(engine)
    finally:
        engine.dispose()

    print(f"Rows loaded: {len(df):,}")

    print("\nSnapshot distribution:")

    snapshot_summary = (
        df.groupby("cutoff_date")
        .agg(
            customers=("customer_id", "count"),
            churned=("churn", "sum"),
        )
        .reset_index()
    )

    snapshot_summary["non_churned"] = (
        snapshot_summary["customers"]
        - snapshot_summary["churned"]
    )

    print(
        snapshot_summary.to_string(
            index=False
        )
    )

    train_df, test_df = split_by_cutoff(df)

    print("\nChronological split:")
    print(
        f"Training cutoffs: "
        f"{train_df['cutoff_date'].min().date()} "
        f"to "
        f"{train_df['cutoff_date'].max().date()}"
    )

    print(
        f"Test cutoff: "
        f"{test_df['cutoff_date'].iloc[0].date()}"
    )

    print(
        f"Training rows: {len(train_df):,}"
    )

    print(
        f"Test rows: {len(test_df):,}"
    )

    X_train, y_train = prepare_model_data(
        train_df,
        feature_columns=DEFAULT_FEATURES,
        target_column="churn",
    )

    X_test, y_test = prepare_model_data(
        test_df,
        feature_columns=DEFAULT_FEATURES,
        target_column="churn",
    )

    print("\nTraining churn distribution:")

    print(
        y_train.value_counts()
        .sort_index()
    )

    print("\nTest churn distribution:")

    print(
        y_test.value_counts()
        .sort_index()
    )

    models = {
        "Logistic Regression":
            build_logistic_regression_model(),

        "Random Forest":
            build_random_forest_model(),
    }

    for model_name, model in models.items():

        print(f"\n{'=' * 60}")
        print(model_name)
        print("Chronological Evaluation")
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
            print(
                f"{metric:12s}: {value:.4f}"
            )

        print("\nConfusion Matrix:")

        print(confusion)

        importance = get_feature_importance(
            trained_model,
            DEFAULT_FEATURES,
        )

        print("\nTop features:")

        print(
            importance.head(10).to_string(
                index=False
            )
        )

        save_metrics(
            metrics,
            model_name,
        )

        save_confusion_matrix(
            confusion,
            model_name,
        )

        save_feature_importance(
            importance,
            model_name,
        )

    print(
        "\nCHRONOLOGICAL MODELING COMPLETED"
    )


if __name__ == "__main__":
    main()