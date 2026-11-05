from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------

RETAIL_FILE = RAW_DATA / "online_retail_II.xlsx"
MARKETING_FILE = RAW_DATA / "marketing_campaign.csv"


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def inspect_dataframe(df: pd.DataFrame, name: str) -> None:
    print("\n" + "=" * 80)
    print(f"{name}")
    print("=" * 80)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    missing_table = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    })

    print(missing_table[missing_table["missing_count"] > 0])

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nFirst 5 rows:")
    print(df.head())


# ---------------------------------------------------------
# Online Retail II
# ---------------------------------------------------------

def inspect_retail() -> None:
    print("\n" + "#" * 80)
    print("ONLINE RETAIL II")
    print("#" * 80)

    if not RETAIL_FILE.exists():
        print(f"ERROR: File not found: {RETAIL_FILE}")
        return

    # Read both sheets
    excel_file = pd.ExcelFile(RETAIL_FILE)

    print("\nExcel sheets:")
    print(excel_file.sheet_names)

    retail_frames = []

    for sheet in excel_file.sheet_names:
        df = pd.read_excel(RETAIL_FILE, sheet_name=sheet)

        print(f"\nSheet: {sheet}")
        print(f"Shape: {df.shape}")

        retail_frames.append(df)

    retail = pd.concat(retail_frames, ignore_index=True)

    inspect_dataframe(
        retail,
        "ONLINE RETAIL II — COMBINED"
    )

    # -----------------------------------------------------
    # Retail-specific checks
    # -----------------------------------------------------

    print("\nRetail-specific analysis:")

    print("\nUnique invoices:")
    print(retail["Invoice"].nunique())

    print("\nUnique customers:")
    print(retail["Customer ID"].nunique())

    print("\nUnique products:")
    print(retail["StockCode"].nunique())

    print("\nUnique countries:")
    print(retail["Country"].nunique())

    print("\nDate range:")

    invoice_dates = pd.to_datetime(
        retail["InvoiceDate"],
        errors="coerce"
    )

    print(f"Min: {invoice_dates.min()}")
    print(f"Max: {invoice_dates.max()}")

    print("\nNegative quantities:")
    print((retail["Quantity"] < 0).sum())

    print("\nZero quantities:")
    print((retail["Quantity"] == 0).sum())

    print("\nNegative prices:")
    print((retail["Price"] < 0).sum())

    print("\nZero prices:")
    print((retail["Price"] == 0).sum())

    print("\nInvoices beginning with 'C':")
    invoice_as_string = retail["Invoice"].astype(str)

    print(
        invoice_as_string
        .str.startswith("C")
        .sum()
    )

    print("\nTop 10 countries:")
    print(
        retail["Country"]
        .value_counts()
        .head(10)
    )

    print("\nCustomer ID missing:")
    print(retail["Customer ID"].isna().sum())


# ---------------------------------------------------------
# Customer Personality / Marketing Campaign
# ---------------------------------------------------------

def inspect_marketing() -> None:
    print("\n" + "#" * 80)
    print("CUSTOMER PERSONALITY / MARKETING CAMPAIGN")
    print("#" * 80)

    if not MARKETING_FILE.exists():
        print(f"ERROR: File not found: {MARKETING_FILE}")
        return

    marketing = pd.read_csv(
        MARKETING_FILE,
        sep="\t"
    )

    inspect_dataframe(
        marketing,
        "CUSTOMER PERSONALITY ANALYSIS"
    )

    # -----------------------------------------------------
    # Marketing-specific checks
    # -----------------------------------------------------

    print("\nMarketing-specific analysis:")

    print("\nUnique customers:")
    print(marketing["ID"].nunique())

    print("\nCustomer ID duplicates:")
    print(marketing["ID"].duplicated().sum())

    print("\nEducation:")
    print(marketing["Education"].value_counts())

    print("\nMarital status:")
    print(marketing["Marital_Status"].value_counts())

    print("\nLatest campaign response:")
    print(marketing["Response"].value_counts())

    print("\nPrevious campaign acceptance:")

    campaign_columns = [
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
    ]

    for column in campaign_columns:
        print(f"\n{column}:")
        print(marketing[column].value_counts())

    print("\nCustomer enrollment date:")

    customer_dates = pd.to_datetime(
        marketing["Dt_Customer"],
        dayfirst=True,
        errors="coerce"
    )

    print(f"Min: {customer_dates.min()}")
    print(f"Max: {customer_dates.max()}")

    print("\nTotal purchases by channel:")

    purchase_columns = [
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ]

    print(
        marketing[purchase_columns].sum()
    )

    print("\nTotal spending columns:")

    spending_columns = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]

    print(
        marketing[spending_columns].sum()
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data directory: {RAW_DATA}")

    inspect_retail()
    inspect_marketing()


if __name__ == "__main__":
    main()