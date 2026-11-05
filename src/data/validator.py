from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw"

RETAIL_FILE = RAW_DATA / "online_retail_II.xlsx"
MARKETING_FILE = RAW_DATA / "marketing_campaign.csv"


def load_retail_data() -> pd.DataFrame:
    """Load and combine both Online Retail II sheets."""

    sheets = pd.read_excel(
        RETAIL_FILE,
        sheet_name=None
    )

    retail = pd.concat(
        sheets.values(),
        ignore_index=True
    )

    return retail


def load_marketing_data() -> pd.DataFrame:
    """Load Customer Personality dataset."""

    return pd.read_csv(
        MARKETING_FILE,
        sep="\t"
    )


def validate_retail_data(df: pd.DataFrame) -> dict:
    """Run structural and domain validation on retail data."""

    results = {}

    expected_columns = {
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    }

    results["required_columns"] = (
        set(df.columns) == expected_columns
    )

    results["row_count"] = len(df)

    results["duplicate_rows"] = int(
        df.duplicated().sum()
    )

    results["missing_customer_ids"] = int(
        df["Customer ID"].isna().sum()
    )

    results["negative_quantities"] = int(
        (df["Quantity"] < 0).sum()
    )

    results["negative_prices"] = int(
        (df["Price"] < 0).sum()
    )

    results["zero_prices"] = int(
        (df["Price"] == 0).sum()
    )

    parsed_dates = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
        format="mixed"
    )

    results["invalid_invoice_dates"] = int(
        parsed_dates.isna().sum()
    )

    return results


def validate_marketing_data(df: pd.DataFrame) -> dict:
    """Run structural and domain validation on marketing data."""

    results = {}

    expected_columns = {
        "ID",
        "Year_Birth",
        "Education",
        "Marital_Status",
        "Income",
        "Kidhome",
        "Teenhome",
        "Dt_Customer",
        "Recency",
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "AcceptedCmp1",
        "AcceptedCmp2",
        "Complain",
        "Z_CostContact",
        "Z_Revenue",
        "Response",
    }

    results["required_columns"] = (
        set(df.columns) == expected_columns
    )

    results["row_count"] = len(df)

    results["duplicate_rows"] = int(
        df.duplicated().sum()
    )

    results["duplicate_customer_ids"] = int(
        df["ID"].duplicated().sum()
    )

    results["missing_income"] = int(
        df["Income"].isna().sum()
    )

    results["invalid_response_values"] = int(
        (~df["Response"].isin([0, 1])).sum()
    )

    results["invalid_campaign_values"] = 0

    campaign_columns = [
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
    ]

    for column in campaign_columns:
        results[f"{column}_invalid"] = int(
            (~df[column].isin([0, 1])).sum()
        )

    return results


def main() -> None:

    print("=" * 80)
    print("DATA VALIDATION")
    print("=" * 80)

    # -----------------------------------------------------
    # Online Retail II
    # -----------------------------------------------------

    retail = load_retail_data()

    retail_results = validate_retail_data(retail)

    print("\nONLINE RETAIL II")

    for key, value in retail_results.items():
        print(f"{key}: {value}")

    # -----------------------------------------------------
    # Customer Personality
    # -----------------------------------------------------

    marketing = load_marketing_data()

    marketing_results = validate_marketing_data(
        marketing
    )

    print("\nCUSTOMER PERSONALITY")

    for key, value in marketing_results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()