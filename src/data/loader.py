from io import StringIO
from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = PROJECT_ROOT / "data" / "raw"


# =========================================================
# Data preparation
# =========================================================

def prepare_retail_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare one Online Retail II sheet for PostgreSQL loading.

    Input:
        Raw Online Retail II DataFrame.

    Output:
        DataFrame matching the retail_transactions table.
    """

    result = df.copy()

    result = result.rename(
        columns={
            "Customer ID": "customer_id",
            "Invoice": "invoice",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "price",
            "Country": "country",
        }
    )

    # Customer ID contains missing values in the raw dataset,
    # so use pandas' nullable integer type.
    result["customer_id"] = (
        pd.to_numeric(
            result["customer_id"],
            errors="coerce",
        )
        .astype("Int64")
    )

    result["quantity"] = (
        pd.to_numeric(
            result["quantity"],
            errors="coerce",
        )
        .astype("Int64")
    )

    result["price"] = pd.to_numeric(
        result["price"],
        errors="coerce",
    )

    result["invoice_date"] = pd.to_datetime(
        result["invoice_date"],
        errors="coerce",
    )

    # Keep exactly the columns expected by PostgreSQL.
    return result[
        [
            "invoice",
            "stock_code",
            "description",
            "quantity",
            "invoice_date",
            "price",
            "customer_id",
            "country",
        ]
    ]


def prepare_marketing_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare Customer Personality Analysis data
    for PostgreSQL loading.

    Input:
        Raw marketing DataFrame.

    Output:
        DataFrame matching the marketing_customers table.
    """

    result = df.copy()

    result = result.rename(
        columns={
            "ID": "customer_id",
            "Year_Birth": "year_birth",
            "Education": "education",
            "Marital_Status": "marital_status",
            "Income": "income",
            "Kidhome": "kidhome",
            "Teenhome": "teenhome",
            "Dt_Customer": "dt_customer",
            "Recency": "recency",
            "MntWines": "mnt_wines",
            "MntFruits": "mnt_fruits",
            "MntMeatProducts": "mnt_meat_products",
            "MntFishProducts": "mnt_fish_products",
            "MntSweetProducts": "mnt_sweet_products",
            "MntGoldProds": "mnt_gold_prods",
            "NumDealsPurchases": "num_deals_purchases",
            "NumWebPurchases": "num_web_purchases",
            "NumCatalogPurchases": "num_catalog_purchases",
            "NumStorePurchases": "num_store_purchases",
            "NumWebVisitsMonth": "num_web_visits_month",
            "AcceptedCmp1": "accepted_cmp1",
            "AcceptedCmp2": "accepted_cmp2",
            "AcceptedCmp3": "accepted_cmp3",
            "AcceptedCmp4": "accepted_cmp4",
            "AcceptedCmp5": "accepted_cmp5",
            "Complain": "complain",
            "Z_CostContact": "z_cost_contact",
            "Z_Revenue": "z_revenue",
            "Response": "response",
        }
    )

    result["customer_id"] = (
        pd.to_numeric(
            result["customer_id"],
            errors="coerce",
        )
        .astype("Int64")
    )

    result["dt_customer"] = pd.to_datetime(
        result["dt_customer"],
        errors="coerce",
        dayfirst=True,
    ).dt.date

    # Keep exactly the columns expected by PostgreSQL.
    return result[
        [
            "customer_id",
            "year_birth",
            "education",
            "marital_status",
            "income",
            "kidhome",
            "teenhome",
            "dt_customer",
            "recency",
            "mnt_wines",
            "mnt_fruits",
            "mnt_meat_products",
            "mnt_fish_products",
            "mnt_sweet_products",
            "mnt_gold_prods",
            "num_deals_purchases",
            "num_web_purchases",
            "num_catalog_purchases",
            "num_store_purchases",
            "num_web_visits_month",
            "accepted_cmp1",
            "accepted_cmp2",
            "accepted_cmp3",
            "accepted_cmp4",
            "accepted_cmp5",
            "complain",
            "z_cost_contact",
            "z_revenue",
            "response",
        ]
    ]


# =========================================================
# PostgreSQL connection
# =========================================================

def get_database_engine():
    """
    Create a SQLAlchemy PostgreSQL engine using .env.

    URL.create() is used instead of manually constructing
    the connection string so special characters such as
    @, #, :, /, etc. in the password are handled safely.
    """

    load_dotenv(PROJECT_ROOT / ".env")

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv(
        "DB_NAME",
        "customer_analytics",
    )
    user = os.getenv(
        "DB_USER",
        "postgres",
    )
    password = os.getenv("DB_PASSWORD")

    if not password:
        raise ValueError(
            "DB_PASSWORD is missing from the .env file."
        )

    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=database,
    )

    return create_engine(connection_url)


# =========================================================
# PostgreSQL COPY helper
# =========================================================

def copy_dataframe_to_postgres(
    df: pd.DataFrame,
    table_name: str,
    engine,
):
    """
    Bulk insert a DataFrame into PostgreSQL using COPY.

    This is substantially more appropriate than repeatedly
    inserting individual rows for large datasets.
    """

    connection = engine.raw_connection()

    try:
        cursor = connection.cursor()

        buffer = StringIO()

        df.to_csv(
            buffer,
            index=False,
            header=False,
            na_rep="\\N",
        )

        buffer.seek(0)

        columns = ", ".join(
            f'"{column}"'
            for column in df.columns
        )

        sql = f"""
            COPY {table_name} ({columns})
            FROM STDIN
            WITH (
                FORMAT CSV,
                NULL '\\N'
            )
        """

        cursor.copy_expert(
            sql,
            buffer,
        )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# =========================================================
# Online Retail II loader
# =========================================================

def load_retail_data(engine):
    """
    Load both Online Retail II Excel sheets
    into retail_transactions.
    """

    file_path = (
        RAW_DATA / "online_retail_II.xlsx"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Retail dataset not found: {file_path}"
        )

    excel_file = pd.ExcelFile(
        file_path
    )

    total_rows = 0

    for sheet_name in excel_file.sheet_names:

        print(
            f"\nLoading sheet: {sheet_name}"
        )

        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
        )

        original_rows = len(df)

        df = prepare_retail_dataframe(df)

        if len(df) != original_rows:
            raise ValueError(
                "Row count changed during retail "
                "data preparation."
            )

        copy_dataframe_to_postgres(
            df=df,
            table_name="retail_transactions",
            engine=engine,
        )

        total_rows += len(df)

        print(
            f"Loaded {len(df):,} rows "
            f"from {sheet_name}"
        )

    print(
        f"\nTotal retail rows loaded: "
        f"{total_rows:,}"
    )

    return total_rows


# =========================================================
# Customer Personality loader
# =========================================================

def load_marketing_data(engine):
    """
    Load Customer Personality Analysis data
    into marketing_customers.
    """

    file_path = (
        RAW_DATA / "marketing_campaign.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Marketing dataset not found: {file_path}"
        )

    # This dataset is tab-separated.
    df = pd.read_csv(
        file_path,
        sep="\t",
    )

    original_rows = len(df)

    df = prepare_marketing_dataframe(df)

    if len(df) != original_rows:
        raise ValueError(
            "Row count changed during marketing "
            "data preparation."
        )

    copy_dataframe_to_postgres(
        df=df,
        table_name="marketing_customers",
        engine=engine,
    )

    print(
        f"Loaded {len(df):,} marketing customers"
    )

    return len(df)


# =========================================================
# Main
# =========================================================

def main():
    """
    Load both datasets into PostgreSQL.
    """

    print(
        "Connecting to customer_analytics..."
    )

    engine = get_database_engine()

    print(
        "Connection configuration loaded."
    )

    retail_rows = load_retail_data(
        engine
    )

    marketing_rows = load_marketing_data(
        engine
    )

    print("\n" + "=" * 60)
    print("DATA LOADING COMPLETED")
    print("=" * 60)
    print(
        f"Retail transactions : {retail_rows:,}"
    )
    print(
        f"Marketing customers  : {marketing_rows:,}"
    )


if __name__ == "__main__":
    main()