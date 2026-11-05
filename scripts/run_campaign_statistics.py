from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from src.statistics.campaign_analysis import (
    chi_square_response_test,
    compare_response_groups,
    summarize_significant_features,
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


def load_marketing_features(engine):
    """Load customer marketing features."""

    query = text(
        """
        SELECT *
        FROM marketing_customer_features
        ORDER BY customer_id
        """
    )

    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def save_results(
    results: pd.DataFrame,
    filename: str,
):
    """Save a dataframe to the results directory."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULTS_DIR / filename,
        index=False,
    )


def main():
    print(
        "Loading marketing customer features..."
    )

    engine = get_database_engine()

    try:
        df = load_marketing_features(engine)
    finally:
        engine.dispose()

    print(
        f"Customers loaded: {len(df):,}"
    )

    # --------------------------------------------------------
    # 1. Categorical analysis
    # --------------------------------------------------------

    print(
        "\nRunning categorical response analysis..."
    )

    categorical_features = [
        "education",
        "marital_status",
    ]

    categorical_results = []

    for feature in categorical_features:

        result = chi_square_response_test(
            df,
            feature,
        )

        categorical_results.append(result)

        print(
            f"\n{feature}"
        )

        print(
            f"Chi-square      : "
            f"{result['chi2']:.4f}"
        )

        print(
            f"p-value         : "
            f"{result['p_value']:.6f}"
        )

        print(
            f"Degrees freedom : "
            f"{result['degrees_of_freedom']}"
        )

    categorical_results_df = pd.DataFrame(
        categorical_results
    )

    save_results(
        categorical_results_df,
        "campaign_categorical_tests.csv",
    )

    # --------------------------------------------------------
    # 2. Numeric behavioral analysis
    # --------------------------------------------------------

    print(
        "\nRunning numeric response analysis..."
    )

    numeric_features = [
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

    numeric_results = compare_response_groups(
        df,
        numeric_features,
    )

    save_results(
        numeric_results,
        "campaign_numeric_tests.csv",
    )

    print(
        "\nNumeric statistical results:"
    )

    print(
        numeric_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 3. Significant numeric features
    # --------------------------------------------------------

    significant_results = (
        summarize_significant_features(
            numeric_results
        )
    )

    save_results(
        significant_results,
        "significant_campaign_features.csv",
    )

    print(
        "\nSignificant numeric features:"
    )

    if significant_results.empty:
        print(
            "No statistically significant "
            "numeric features found."
        )
    else:
        print(
            significant_results.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 4. Education response rates
    # --------------------------------------------------------

    education_summary = (
        df.groupby("education")
        .agg(
            customers=("customer_id", "count"),
            responders=("response", "sum"),
            response_rate=("response", "mean"),
            average_spend=("total_spend", "mean"),
        )
        .reset_index()
    )

    education_summary["response_rate"] *= 100

    save_results(
        education_summary,
        "campaign_response_by_education.csv",
    )

    # --------------------------------------------------------
    # 5. Customer value quartiles
    # --------------------------------------------------------

    value_df = df.copy()

    value_df["spend_quartile"] = pd.qcut(
        value_df["total_spend"],
        q=4,
        labels=False,
        duplicates="drop",
    ) + 1

    value_summary = (
        value_df.groupby("spend_quartile")
        .agg(
            customers=("customer_id", "count"),
            responders=("response", "sum"),
            response_rate=("response", "mean"),
            average_spend=("total_spend", "mean"),
            average_purchases=(
                "total_purchases",
                "mean",
            ),
        )
        .reset_index()
    )

    value_summary["response_rate"] *= 100

    save_results(
        value_summary,
        "campaign_response_by_value.csv",
    )

    print(
        "\nCustomer value response analysis:"
    )

    print(
        value_summary.to_string(
            index=False
        )
    )

    print(
        "\nCAMPAIGN STATISTICAL ANALYSIS COMPLETED"
    )


if __name__ == "__main__":
    main()