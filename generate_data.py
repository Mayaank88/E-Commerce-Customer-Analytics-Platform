"""
Synthetic E-Commerce Dataset Generator
======================================
Generates a realistic, warehouse-ready e-commerce dataset with 10,000+ records:

  * Multi-item shopping baskets (a transaction may contain 1-3 line items),
    so Market Basket Analysis has genuine item co-occurrence to exploit
  * Category affinity: items inside a basket are often complementary
    (Electronics<->Toys, Clothing<->Beauty, ...), producing lift > 1 rules
  * Customer demographics, loyalty tiers, channels, payment methods
  * RFM (Recency / Frequency / Monetary) metrics per customer
  * Supervised labels: is_high_value, is_churn_risk and a CLV proxy

Outputs (written next to this script):
    ecommerce_10k.csv          - main fact table (transaction grain, >= 10,000 rows)
    customer_rfm_summary.csv   - customer-grain RFM table
    customer_demographics.csv  - customer dimension table

Usage:
    python generate_data.py

Deterministic: fixed random seeds -> identical output on every run.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)

# --------------------------------------------------------------------------------
# Domain configuration
# --------------------------------------------------------------------------------
CATEGORIES = [
    "Electronics", "Clothing", "Home & Garden", "Sports", "Books",
    "Beauty", "Toys", "Automotive", "Health", "Food & Beverage",
]
CATEGORY_WEIGHTS = [0.20, 0.18, 0.12, 0.10, 0.08, 0.10, 0.07, 0.05, 0.05, 0.05]

# Complementary categories frequently bought together inside one basket.
# Used to create genuine association-rule signal (lift > 1).
COMPLEMENTARY = {
    "Electronics": ["Toys", "Automotive", "Sports"],
    "Clothing": ["Beauty", "Sports", "Home & Garden"],
    "Home & Garden": ["Books", "Food & Beverage", "Sports"],
    "Sports": ["Clothing", "Health", "Electronics"],
    "Books": ["Home & Garden", "Toys", "Food & Beverage"],
    "Beauty": ["Health", "Clothing", "Food & Beverage"],
    "Toys": ["Electronics", "Books", "Sports"],
    "Automotive": ["Electronics", "Health", "Sports"],
    "Health": ["Beauty", "Food & Beverage", "Sports"],
    "Food & Beverage": ["Health", "Home & Garden", "Books"],
}

PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Apple Pay",
                   "Google Pay", "Bank Transfer"]
PAYMENT_WEIGHTS = [0.35, 0.25, 0.20, 0.10, 0.05, 0.05]

CHANNELS = ["Website", "Mobile App", "In-Store", "Marketplace"]
CHANNEL_WEIGHTS = [0.45, 0.35, 0.15, 0.05]

REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America",
           "Middle East & Africa"]
REGION_WEIGHTS = [0.35, 0.30, 0.20, 0.10, 0.05]

# Price multipliers per category and per customer value tier
CATEGORY_PRICE_MULTIPLIER = {
    "Electronics": 3.0, "Clothing": 1.0, "Home & Garden": 1.5,
    "Sports": 1.8, "Books": 0.5, "Beauty": 0.8,
    "Toys": 0.7, "Automotive": 2.5, "Health": 1.2, "Food & Beverage": 0.6,
}
TIER_MULTIPLIER = {"low": 0.6, "medium": 1.0, "high": 1.8, "vip": 3.0}

# Small product pool so that individual products repeat across baskets
# (essential for product-level association mining).
PRODUCTS_PER_CATEGORY = 40


def _build_product_pool():
    """Create a fixed catalogue: ~40 named products per category."""
    return {cat: [f"{cat} Product {i:03d}" for i in range(1, PRODUCTS_PER_CATEGORY + 1)]
            for cat in CATEGORIES}


def generate_synthetic_data(n_records=12000, n_customers=2500):
    """
    Generate a synthetic e-commerce dataset with realistic patterns.

    Parameters
    ----------
    n_records : int
        Target number of *line-item* rows (>= 10,000).
    n_customers : int
        Number of unique customers in the customer dimension.

    Returns
    -------
    (transactions_df, rfm_df, customers_df)
    """
    product_pool = _build_product_pool()

    # --- Customer dimension ---------------------------------------------------
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
    customers_df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": np.random.normal(38, 12, n_customers).astype(int).clip(18, 80),
        "gender": np.random.choice(["Male", "Female", "Other"],
                                   n_customers, p=[0.48, 0.48, 0.04]),
        "region": np.random.choice(REGIONS, n_customers, p=REGION_WEIGHTS),
        "signup_date": [datetime(2020, 1, 1) + timedelta(days=int(np.random.randint(0, 1000)))
                        for _ in range(n_customers)],
        "loyalty_tier": np.random.choice(["Bronze", "Silver", "Gold", "Platinum"],
                                         n_customers, p=[0.50, 0.30, 0.15, 0.05]),
        "email_subscribed": np.random.choice([True, False], n_customers, p=[0.70, 0.30]),
        "marketing_opt_in": np.random.choice([True, False], n_customers, p=[0.60, 0.40]),
    })

    # --- Customer behaviour profiles -----------------------------------------
    profiles = {}
    for cust_id in customer_ids:
        profiles[cust_id] = {
            "value_tier": np.random.choice(["low", "medium", "high", "vip"],
                                           p=[0.40, 0.35, 0.20, 0.05]),
            "avg_order_value": np.random.lognormal(3.5, 0.8),
            "category_preference": np.random.choice(CATEGORIES, p=CATEGORY_WEIGHTS),
        }

    # --- Basket generation ----------------------------------------------------
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range_days = (end_date - start_date).days

    # Target basket count: basket sizes 1..3 items with mean ~1.55
    basket_sizes = np.random.choice([1, 2, 3], size=n_records, p=[0.58, 0.29, 0.13])
    n_baskets = int(np.ceil(n_records / basket_sizes.mean()))

    rows = []
    txn_seq = 0
    lines_written = 0

    while lines_written < n_records and txn_seq < n_baskets:
        txn_seq += 1
        customer_id = customer_ids[int(np.random.randint(0, n_customers))]
        profile = profiles[customer_id]
        n_items = int(basket_sizes[(txn_seq - 1) % len(basket_sizes)])

        # One date per basket, with Nov/Dec holiday seasonality
        txn_date = start_date + timedelta(days=int(np.random.randint(0, date_range_days)))
        month = txn_date.month
        seasonal = 1.5 if month in (11, 12) else (0.7 if month in (1, 2) else 1.0)

        channel = np.random.choice(CHANNELS, p=CHANNEL_WEIGHTS)
        payment = np.random.choice(PAYMENT_METHODS, p=PAYMENT_WEIGHTS)
        txn_id = f"TXN_{txn_seq:08d}"

        # First item follows the customer's category preference 60% of the time
        if np.random.random() < 0.6:
            first_cat = profile["category_preference"]
        else:
            first_cat = np.random.choice(CATEGORIES, p=CATEGORY_WEIGHTS)

        chosen_cats = [first_cat]
        for _ in range(1, n_items):
            # 65%: pull a complementary category from something already in the basket
            if np.random.random() < 0.65:
                options = []
                for c in chosen_cats:
                    options.extend(COMPLEMENTARY.get(c, []))
                options = [o for o in options if o not in chosen_cats] or \
                          [c for c in CATEGORIES if c not in chosen_cats]
                nxt = options[int(np.random.randint(0, len(options)))]
            else:
                nxt = np.random.choice(CATEGORIES, p=CATEGORY_WEIGHTS)
            chosen_cats.append(nxt)

        for category in chosen_cats:
            if lines_written >= n_records:
                break
            lines_written += 1

            base_price = (profile["avg_order_value"]
                          * CATEGORY_PRICE_MULTIPLIER[category]
                          * TIER_MULTIPLIER[profile["value_tier"]]
                          * seasonal)
            quantity = max(1, int(np.random.poisson(1.5)))
            unit_price = max(5.0, base_price * np.random.lognormal(0, 0.3))
            total_amount = round(unit_price * quantity, 2)

            # 30% chance of a discount
            discount_pct = 0
            if np.random.random() < 0.3:
                discount_pct = int(np.random.choice([5, 10, 15, 20, 25],
                                                    p=[0.4, 0.3, 0.15, 0.1, 0.05]))
            discount_amount = round(total_amount * discount_pct / 100, 2)
            final_amount = round(total_amount - discount_amount, 2)

            rows.append({
                "transaction_id": txn_id,
                "customer_id": customer_id,
                "transaction_date": txn_date,
                "category": category,
                "product_name": product_pool[category][int(np.random.randint(
                    0, PRODUCTS_PER_CATEGORY))],
                "quantity": quantity,
                "unit_price": round(unit_price, 2),
                "total_amount": total_amount,
                "discount_pct": discount_pct,
                "discount_amount": discount_amount,
                "final_amount": final_amount,
                "payment_method": payment,
                "channel": channel,
            })

    transactions_df = pd.DataFrame(rows)

    # --- RFM metrics per customer (frequency = distinct baskets) --------------
    snapshot_date = pd.Timestamp("2025-01-01")

    rfm = transactions_df.groupby("customer_id").agg(
        last_purchase_date=("transaction_date", "max"),
        frequency=("transaction_id", "nunique"),           # number of baskets
        monetary=("final_amount", "sum"),
        avg_order_value=("final_amount", "mean"),
        total_quantity=("quantity", "sum"),
        unique_categories=("category", "nunique"),
        preferred_channel=("channel",
                           lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"),
        preferred_payment=("payment_method",
                           lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"),
    ).reset_index()

    rfm["recency"] = (snapshot_date - rfm["last_purchase_date"]).dt.days

    signup_dates = customers_df.set_index("customer_id")["signup_date"]
    rfm["signup_date"] = rfm["customer_id"].map(signup_dates)
    rfm["tenure_days"] = (snapshot_date - rfm["signup_date"]).dt.days

    # --- Supervised labels ----------------------------------------------------
    # High-value: top 20% monetary AND top 20% frequency
    mono_cut = rfm["monetary"].quantile(0.80)
    freq_cut = rfm["frequency"].quantile(0.80)
    rfm["is_high_value"] = ((rfm["monetary"] >= mono_cut) &
                            (rfm["frequency"] >= freq_cut)).astype(int)

    # Churn risk: stale recency, or stale + below-median frequency
    rfm["is_churn_risk"] = ((rfm["recency"] > 90) |
                            ((rfm["recency"] > 60) &
                             (rfm["frequency"] < rfm["frequency"].median()))).astype(int)

    # CLV proxy (see PROJECT_REPORT.md Eq. 8), winsorised at P99
    clv = (rfm["monetary"] * (1 + 1 / rfm["recency"].clip(1)) *
           (rfm["frequency"] / rfm["tenure_days"].clip(1)) * 365)
    rfm["clv_proxy"] = clv.clip(0, clv.quantile(0.99)).round(2)

    # --- Assemble the denormalised fact table --------------------------------
    final_df = transactions_df.merge(customers_df, on="customer_id", how="left")
    final_df = final_df.merge(
        rfm[["customer_id", "recency", "frequency", "monetary", "avg_order_value",
             "total_quantity", "unique_categories", "preferred_channel",
             "preferred_payment", "tenure_days", "is_high_value", "is_churn_risk",
             "clv_proxy"]],
        on="customer_id", how="left")

    final_df["days_since_signup"] = (final_df["transaction_date"]
                                     - final_df["signup_date"]).dt.days
    final_df["is_weekend"] = final_df["transaction_date"].dt.dayofweek.isin([5, 6]).astype(int)
    final_df["month"] = final_df["transaction_date"].dt.month
    final_df["quarter"] = final_df["transaction_date"].dt.quarter
    final_df["year"] = final_df["transaction_date"].dt.year

    column_order = [
        "transaction_id", "customer_id", "transaction_date", "year", "quarter", "month",
        "is_weekend", "days_since_signup",
        "category", "product_name", "quantity", "unit_price", "total_amount",
        "discount_pct", "discount_amount", "final_amount",
        "payment_method", "channel", "region",
        "age", "gender", "signup_date", "loyalty_tier",
        "email_subscribed", "marketing_opt_in",
        "recency", "frequency", "monetary", "avg_order_value", "total_quantity",
        "unique_categories", "preferred_channel", "preferred_payment", "tenure_days",
        "is_high_value", "is_churn_risk", "clv_proxy",
    ]
    final_df = final_df[column_order]

    return final_df, rfm, customers_df


def main():
    """Generate all three CSV artefacts and print a summary report."""
    out_dir = os.path.dirname(os.path.abspath(__file__))
    n_records, n_customers = 12000, 2500

    print("Generating synthetic e-commerce dataset ...")
    print(f"Target: ~{n_records:,} line items across multi-item baskets, "
          f"{n_customers:,} customers\n")

    df, rfm, customers = generate_synthetic_data(n_records=n_records,
                                                 n_customers=n_customers)

    # --- Main fact table ------------------------------------------------------
    fact_path = os.path.join(out_dir, "ecommerce_10k.csv")
    df.to_csv(fact_path, index=False)
    print(f"Main dataset saved to: {fact_path}")
    print(f"  Shape          : {df.shape}")
    print(f"  Unique baskets : {df['transaction_id'].nunique():,}")
    print(f"  Unique customers: {df['customer_id'].nunique():,}")
    print(f"  Date range     : {df['transaction_date'].min()} .. {df['transaction_date'].max()}")
    print(f"  High-value     : {df['is_high_value'].sum():,} "
          f"({df['is_high_value'].mean() * 100:.1f}% of rows)")
    print(f"  Churn risk     : {df['is_churn_risk'].sum():,} "
          f"({df['is_churn_risk'].mean() * 100:.1f}% of rows)")
    assert len(df) >= 10_000, "Dataset must contain at least 10,000 records"

    # --- Customer RFM summary -------------------------------------------------
    rfm_path = os.path.join(out_dir, "customer_rfm_summary.csv")
    rfm.to_csv(rfm_path, index=False)
    print(f"\nRFM summary saved to: {rfm_path}  |  shape {rfm.shape}")

    # --- Customer dimension ---------------------------------------------------
    cust_path = os.path.join(out_dir, "customer_demographics.csv")
    customers.to_csv(cust_path, index=False)
    print(f"Customer demographics saved to: {cust_path}  |  shape {customers.shape}")

    # --- Summary statistics ---------------------------------------------------
    basket_sizes = df.groupby("transaction_id").size()
    print("\n" + "=" * 64)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 64)
    print(f"\nBasket size distribution:\n{basket_sizes.value_counts().sort_index()}")
    print(f"\nTransaction amount:\n{df['final_amount'].describe()}")
    print(f"\nCategory distribution:\n{df['category'].value_counts()}")
    print(f"\nLoyalty tier:\n{df['loyalty_tier'].value_counts()}")
    print(f"\nChannel mix:\n{df['channel'].value_counts()}")
    print(f"\nPayment method:\n{df['payment_method'].value_counts()}")

    return df, rfm, customers


if __name__ == "__main__":
    main()
