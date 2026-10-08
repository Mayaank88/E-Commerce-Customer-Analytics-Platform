"""
prepare_online_retail.py
========================
Convert the Kaggle / UCI "Online Retail" spreadsheet (the classic e-commerce
dataset, Dec 2010 - Dec 2011, UK retailer) into the canonical warehouse schema
used by `app.py` and store it as `online_retail.csv`.

The source workbook ships as:
    online+retail.zip  ->  "Online Retail.xlsx"  (541,909 line-items, 8 cols)

Standard data-quality preprocessing applied during conversion (documented in
PROJECT_REPORT.md §2 / §8):

  1. Drop cancellation invoices   (InvoiceNo starting with 'C')   -9,288
  2. Drop line items with no CustomerID (cannot join a basket)   -135,080
  3. Drop invalid sales           (Quantity <= 0 or UnitPrice <= 0) = returns,
     freebies and price-entry artefacts                          -~10k
  4. Derive `amount = Quantity x UnitPrice`, map columns onto the
     application's canonical schema
  5. Dump `online_retail.csv` (~400k clean records, ~17 MB)

Usage:
    python prepare_online_retail.py --src "/path/to/Online Retail.xlsx"
    python prepare_online_retail.py --src online+retail.zip       # zip is accepted

The output file is intentionally exported (not parsed at runtime) so that the
Streamlit app stays dependency-light (no openpyxl needed in production).
"""

import argparse
import os
import zipfile

import pandas as pd


# ----------------------------------------------------------------------------
# Canonical output
# ----------------------------------------------------------------------------
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "online_retail.csv")

CANONICAL_COLUMNS = [
    "transaction_id",   # INV_<original InvoiceNo>
    "customer_id",      # CUST_<padded CustomerID>
    "transaction_date", # original InvoiceDate (datetime)
    "stock_code",       # original StockCode
    "product_name",     # Description (uppercased, fallback = StockCode)
    "quantity",         # original Quantity (>0)
    "unit_price",       # original UnitPrice (>0)
    "amount",           # Quantity x UnitPrice  (money column used by the app)
    "region",           # original Country (dimension attribute)
]


def resolve_source(src: str) -> str:
    """Return a path to the .xlsx, extracting a zip if needed."""
    if src.lower().endswith(".zip"):
        zdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".online_retail_src")
        os.makedirs(zdir, exist_ok=True)
        with zipfile.ZipFile(src) as z:
            members = [m for m in z.namelist() if m.lower().endswith(".xlsx")]
            if not members:
                raise FileNotFoundError("No .xlsx inside the zip archive.")
            z.extract(members[0], zdir)
        return os.path.join(zdir, members[0])
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source not found: {src}")
    return src


def convert(src: str, out: str = OUT_PATH) -> None:
    """Read the workbook, clean it, and export the canonical CSV."""
    xlsx_path = resolve_source(src)
    print(f"[1/5] Reading {xlsx_path} ...")
    df = pd.read_excel(xlsx_path)
    n0 = len(df)
    log: list[str] = []

    # --- 1. Cancellations -----------------------------------------------------
    is_cancel = df["InvoiceNo"].astype(str).str.strip().str.upper().str.startswith("C")
    n_cancel = int(is_cancel.sum())
    df = df[~is_cancel]
    log.append(f"Cancellation invoices removed        : {n_cancel:>9,} rows")
    df = df[df["CustomerID"].isin([0.0]) == False]  # noqa: E712 — guard zero IDs too

    # --- 2. Missing customer ---------------------------------------------------
    n_missing_cust = int(df["CustomerID"].isna().sum())
    df = df[df["CustomerID"].notna()]
    log.append(f"Missing CustomerID dropped           : {n_missing_cust:>9,} rows")

    # --- 3. Invalid sales ------------------------------------------------------
    valid_sale = (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
    n_invalid = int((~valid_sale).sum())
    df = df[valid_sale]
    log.append(f"Invalid quantity / price dropped     : {n_invalid:>9,} rows")

    # --- 4. Canonical schema ---------------------------------------------------
    lon = pd.to_datetime(df["InvoiceDate"])
    df["customer_id"] = "CUST_" + df["CustomerID"].astype(int).astype(str).str.zfill(6)
    df["transaction_id"] = "INV_" + df["InvoiceNo"].astype(str).str.strip()
    df["transaction_date"] = lon
    df["stock_code"] = df["StockCode"].astype(str).str.strip()
    df["product_name"] = df["Description"].fillna(df["stock_code"]).str.strip().str.upper()
    df["quantity"] = df["Quantity"].astype(int)
    df["unit_price"] = df["UnitPrice"].round(2)
    df["amount"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    df["region"] = df["Country"].astype(str)

    out_df = df[["transaction_id", "customer_id", "transaction_date", "stock_code",
                 "product_name", "quantity", "unit_price", "amount", "region"]]

    # --- 5. Export --------------------------------------------------------------
    out_df.to_csv(out, index=False)

    print("[2/5] Cleaning summary:")
    for line in log:
        print(f"        {line}")
    print(f"        Remaining records                : {len(out_df):>9,} "
          f"(kept {100 * len(out_df) / n0:.1f}% of {n0:,})")
    print(f"[3/5] Invoices  : {out_df['transaction_id'].nunique():,}")
    print(f"[4/5] Customers : {out_df['customer_id'].nunique():,}")
    print(f"[5/5] Saved     : {out}")
    print(f"        Date range: {out_df['transaction_date'].min()} .. "
          f"{out_df['transaction_date'].max()}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        help="Path to 'Online Retail.xlsx' or the downloaded online+retail.zip",
        default="online+retail.zip",
    )
    args = parser.parse_args()
    convert(args.src)