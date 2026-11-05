from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import text

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import get_database_engine
from src.statistics.churn_analysis import (
    compare_churn_groups,
    summarize_significant_features,
)


FEATURE_COLUMNS = [
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


def load_model_dataset(engine):
    """Load the churn modeling dataset from PostgreSQL."""

    query = text("""
        SELECT *
        FROM churn_model_dataset
    """)

    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def main():
    print("Connecting to PostgreSQL...")

    engine = get_database_engine()

    print("Loading churn modeling dataset...")

    df = load_model_dataset(engine)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    print("\nChurn distribution:")
    print(
        df["churn"]
        .value_counts()
        .sort_index()
    )

    print("\nRunning Mann-Whitney U tests...")

    results = compare_churn_groups(
        df,
        FEATURE_COLUMNS,
    )

    results = results.sort_values(
        "adjusted_p_value"
    ).reset_index(drop=True)

    significant = summarize_significant_features(
        results
    )

    print("\nStatistical Results:")
    print(
        results.to_string(
            index=False
        )
    )

    print("\nSignificant Features:")
    print(
        significant.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(
        exist_ok=True
    )

    results_path = (
        results_dir
        / "churn_statistical_tests.csv"
    )

    significant_path = (
        results_dir
        / "significant_churn_features.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    significant.to_csv(
        significant_path,
        index=False,
    )

    print(
        f"\nSaved full results to:"
        f"\n{results_path}"
    )

    print(
        f"\nSaved significant features to:"
        f"\n{significant_path}"
    )


if __name__ == "__main__":
    main()