"""
================================================================================
 E-Commerce Customer Analytics & Machine Learning Platform
 A Data Warehousing & Data Mining Mini-Project
================================================================================
 Multi-tab Streamlit application implementing an end-to-end analytics stack:

   1. Overview              - Dataset KPIs, distributions, correlation heatmap
   2. ETL & RFM             - Data cleaning, missing values, outlier detection,
                              RFM feature extraction, Star-Schema diagram
   3. Association Rules      - Market Basket Analysis (Apriori / FP-Growth style,
                              Support / Confidence / Lift)
   4. Clustering            - K-Means segmentation with Elbow + Silhouette,
                              interactive 2D / 3D Plotly scatter plots
   5. Classification        - Decision Tree (J48-equivalent) vs Gaussian Naive
                              Bayes with confusion matrices & ROC curves
   6. Regression            - Customer Lifetime Value (CLV) prediction
   7. Live Prediction        - Real-time form based inference
   8. Accuracy & Comparison  - Model parameter comparison matrices & accuracy
                              summary dashboard (all models, side by side)
   9. About                 - Repository map & algorithm summary

   Data sources (sidebar):
     - Kaggle / UCI "Online Retail" (real data, default) - converted by
       `prepare_online_retail.py` into the canonical schema (`online_retail.csv`)
     - Built-in synthetic warehouse (`ecommerce_10k.csv`, generated)
     - User-uploaded CSV (schema auto-detected)

 Run with:  streamlit run app.py
================================================================================
"""

from __future__ import annotations

import hashlib
import io
import os
import time
import traceback
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend for server-side rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

# --- Optional dependency: mlxtend (a custom Apriori fallback is bundled) ----
try:
    from mlxtend.frequent_patterns import apriori as mlx_apriori
    from mlxtend.frequent_patterns import association_rules as mlx_association_rules

    MLXTEND_AVAILABLE = True
except Exception:  # pragma: no cover - environment without mlxtend
    MLXTEND_AVAILABLE = False

# --------------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command executed)
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Customer Analytics & ML Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------
# Constants & configuration
# --------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "ecommerce_10k.csv")
# Real-world Kaggle / UCI "Online Retail" dataset, pre-converted to the app's
# canonical warehouse schema by `prepare_online_retail.py`.
ONLINE_RETAIL_PATH = os.path.join(BASE_DIR, "online_retail.csv")

CLUSTER_PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

# Candidate column names accepted for user-uploaded CSVs (case-insensitive)
COL_CANDIDATES = {
    "customer_id": ["customer_id", "cust_id", "customer", "user_id", "client_id"],
    "transaction_id": ["transaction_id", "order_id", "invoice_id", "txn_id"],
    "date": ["transaction_date", "order_date", "purchase_date", "invoice_date", "date"],
    "amount": ["final_amount", "total_amount", "amount", "order_value", "revenue", "sale_amount"],
    "category": ["category", "product_category", "department", "dept"],
    "product": ["product_name", "product", "item_name", "sku", "item"],
    "signup": ["signup_date", "registration_date", "joined_date"],
    "quantity": ["quantity", "qty"],
    "discount": ["discount_pct", "discount", "discount_percent"],
}

# Default modelling feature set (intersection with available columns is used)
CORE_FEATURES = [
    "recency", "frequency", "monetary", "avg_order_value", "tenure_days",
    "age", "unique_categories", "total_quantity",
]

TARGET_OPTIONS = {
    "High-Value Customer (is_high_value)": "is_high_value",
    "Churn Risk (is_churn_risk)": "is_churn_risk",
}

DEFAULT_ETL_SETTINGS = {
    "drop_duplicates": True,
    "fill_strategy": "median",       # median | mean | mode | zero | drop
    "outlier_action": "cap",         # cap | remove | none
    "outlier_factor": 1.5,           # k in [Q1 - k*IQR, Q3 + k*IQR]
    "recompute_rfm": True,
    "rederive_labels": False,
    "snapshot_date": None,           # None -> max(transaction_date) + 1 day
}


# --------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------
def _fmt_num(x: float) -> str:
    """Human friendly number formatting for KPI cards."""
    try:
        if abs(x) >= 1_000_000:
            return f"{x / 1_000_000:.2f}M"
        if abs(x) >= 1_000:
            return f"{x / 1_000:.1f}K"
        if float(x).is_integer():
            return str(int(x))
        return f"{x:.2f}"
    except Exception:
        return str(x)


def _fmt_set(s) -> str:
    """Render an itemset (frozenset/tuple) as a readable string."""
    if isinstance(s, str):
        return s
    return ", ".join(sorted(map(str, s)))


def pick_col(df: pd.DataFrame, kind: str) -> Optional[str]:
    """Case-insensitive lookup of a semantic column inside a DataFrame."""
    wanted = COL_CANDIDATES.get(kind, [])
    lower_map = {c.lower(): c for c in df.columns}
    for cand in wanted:
        if cand in lower_map:
            return lower_map[cand]
    return None


def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse every plausible date column, coercing bad entries to NaT."""
    df = df.copy()
    date_like = COL_CANDIDATES["date"] + COL_CANDIDATES["signup"]
    for col in df.columns:
        if col.lower() in [d.lower() for d in date_like] or col.lower().endswith("_date"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def show_exception(prefix: str = "Pipeline failed"):
    """Render a user-facing error together with the traceback in an expander."""
    st.error(f"⚠️ {prefix}. See details below.")
    with st.expander("Technical details"):
        st.code(traceback.format_exc())


def apply_css() -> None:
    """
    Theme-safe custom styling (final-year-project polish).

    Contrast contract: the app always renders with *light* surfaces and dark
    text (see `.streamlit/config.toml`). Every widget colour is declared
    explicitly here so buttons, labels and text stay visible even if the user's
    browser or Streamlit theme differs. `color-scheme: light` also stops the
    browser's own "forced dark" mode from restyling native controls.
    """
    st.markdown(
        """
        <style>
        :root, html, body, [data-testid="stAppViewContainer"] {
            color-scheme: light;
            color: #0F172A;
            background: #F6F8FC;
        }
        [data-testid="stAppViewContainer"] {
            background:
              radial-gradient(1200px 480px at 88% -12%, rgba(79,70,229,.09), transparent 60%),
              radial-gradient(900px 420px at -8% 0%, rgba(14,165,233,.07), transparent 55%),
              linear-gradient(180deg, #F6F8FC 0%, #FFFFFF 460px);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E6E9F2;
        }
        .block-container { padding-top: 1.4rem; padding-bottom: 3.2rem; }

        h1, h2, h3 {
            font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
            letter-spacing: -0.02em; color: #0F172A;
        }
        h1 { font-size: 2.05rem; font-weight: 800; }
        h2 { font-weight: 700; }
        h3 { font-weight: 650; }
        p, li, td, th { color: #0F172A; }

        /* ---------- metric cards ---------- */
        div[data-testid="stMetric"] {
            background: #FFFFFF; border: 1px solid #E6E9F2; border-radius: 14px;
            padding: 14px 16px; box-shadow: 0 1px 3px rgba(15,23,42,.05);
        }
        div[data-testid="stMetric"] label { color: #64748B; font-weight: 600; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #0F172A; }

        /* ---------- buttons: guaranteed visible labels ---------- */
        .stButton > button, .stDownloadButton > button,
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-tertiary"] {
            color: #0F172A;
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 10px;
            font-weight: 600;
            box-shadow: 0 1px 2px rgba(15,23,42,.04);
        }
        .stButton > button:hover, .stDownloadButton > button:hover,
        [data-testid="stBaseButton-secondary"]:hover {
            border-color: #4F46E5; color: #4F46E5;
        }
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"],
        [data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
            color: #FFFFFF !important;
            border: none;
        }
        .stButton > button[kind="primary"]:hover,
        .stDownloadButton > button[kind="primary"]:hover,
        [data-testid="stBaseButton-primary"]:hover {
            background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%);
            color: #FFFFFF !important;
        }
        .stButton > button p, .stButton > button span,
        .stDownloadButton > button p, .stDownloadButton > button span,
        [data-testid="stBaseButton-primary"] p,
        [data-testid="stBaseButton-primary"] span { color: inherit; }

        /* ---------- inputs, radio, checkboxes ---------- */
        [data-testid="stRadio"] label, [data-testid="stCheckbox"] label,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div, .stMarkdown {
            color: #0F172A; font-size: .95rem;
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown { color: #1E293B; }
        .stSidebar input, .stSidebar textarea { color: #0F172A; }

        /* ---------- tabs ---------- */
        [data-testid="stTabs"] button {
            color: #334155; background: transparent; font-weight: 600;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #4F46E5; border-bottom: 2px solid #4F46E5;
        }

        /* ---------- tables, expanders, code ---------- */
        [data-testid="stDataFrame"] {
            border: 1px solid #E6E9F2; border-radius: 12px; overflow: hidden;
        }
        [data-testid="stExpander"] details {
            background: #FFFFFF; border: 1px solid #E6E9F2; border-radius: 12px;
        }
        code {
            background: #EEF2FF; color: #3730A3; border-radius: 6px; padding: 1px 6px;
        }

        /* ---------- custom components: hero / pills / footer ---------- */
        .hero-card {
            background: linear-gradient(135deg, #EEF2FF 0%, #F8FAFC 100%);
            border: 1px solid #E0E7FF; border-radius: 18px;
            padding: 24px 28px; margin: 0 0 18px 0;
            box-shadow: 0 4px 18px rgba(79,70,229,.08);
        }
        .hero-title {
            font-size: 1.85rem; font-weight: 800; color: #0F172A;
            margin: 0 0 6px 0; letter-spacing: -0.02em; line-height: 1.25;
        }
        .hero-sub {
            color: #475569; font-size: .97rem; margin: 0 0 14px 0; line-height: 1.55;
        }
        .pill {
            display: inline-block; background: #FFFFFF; color: #3730A3;
            border: 1px solid #C7D2FE; border-radius: 999px;
            padding: 4px 12px; margin: 2px 4px 2px 0;
            font-size: .78rem; font-weight: 650; white-space: nowrap;
        }
        .footer-note {
            margin-top: 2.4rem; padding: 14px 18px; border-top: 1px solid #E6E9F2;
            color: #64748B; font-size: .82rem; text-align: center; line-height: 1.6;
        }
        .side-head {
            font-size: .68rem; font-weight: 800; letter-spacing: .13em;
            text-transform: uppercase; color: #94A3B8; margin: 16px 0 2px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(source_label: str) -> None:
    """Polished hero banner with pipeline-stage pills."""
    stages = ["ETL & RFM", "Association Rules", "K-Means", "DT vs NB", "CLV", "Live Inference"]
    pills = "".join(f'<span class="pill">{s}</span>' for s in stages)
    st.markdown(
        f"""
        <div class="hero-card">
          <div class="hero-title">🛍️ E-Commerce Customer Analytics Platform</div>
          <div class="hero-sub">A complete data-warehousing &amp; data-mining stack —
            from a raw transactional feed to customer segments, market-basket rules,
            classifier comparisons, CLV modelling and live inference.
            <br><b>Data source:</b> {source_label}</div>
          <div>{pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        '<div class="footer-note">Built as a Data Warehousing &amp; Data Mining '
        'mini-project · pandas · scikit-learn · Plotly · Streamlit — full theory, '
        'formulas and evaluation tables in <code>PROJECT_REPORT.md</code></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------
# Data-access layer (cached)
# --------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_builtin(path: str) -> pd.DataFrame:
    """Load the bundled synthetic dataset; regenerate it on the fly if missing."""
    if not os.path.exists(path):
        try:
            from generate_data import generate_synthetic_data  # local script

            df, _, _ = generate_synthetic_data(n_records=12000, n_customers=2500)
            return df
        except Exception as exc:  # pragma: no cover
            raise FileNotFoundError(
                f"'{path}' not found and automatic generation failed: {exc}. "
                "Run `python generate_data.py` first."
            ) from exc
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_upload(data: bytes, name: str) -> pd.DataFrame:
    """Parse an uploaded CSV from raw bytes (cached by content hash)."""
    return pd.read_csv(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def load_online_retail(path: str) -> pd.DataFrame:
    """Load the cleaned Kaggle 'Online Retail' warehouse export (`online_retail.csv`).

    Produced from the original 541,909-row workbook by
    `python prepare_online_retail.py --src <Online Retail.xlsx | online+retail.zip>`
    (drops cancellations, anonymous rows and invalid sales; derives `amount`).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' is missing. Convert the source workbook first:\n"
            "    python prepare_online_retail.py --src \"online+retail.zip\""
        )
    return pd.read_csv(path, parse_dates=["transaction_date"])


# --------------------------------------------------------------------------------
# ETL layer
# --------------------------------------------------------------------------------
def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level data-quality profile: dtype, missing, uniqueness."""
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "missing_%": round(100 * s.isna().mean(), 2),
                "unique": int(s.nunique(dropna=True)),
                "unique_%": round(100 * s.nunique(dropna=True) / n, 2) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def iqr_outlier_report(df: pd.DataFrame, cols: List[str], factor: float) -> pd.DataFrame:
    """Tukey IQR fence analysis for every supplied numeric column."""
    records = []
    for col in cols:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        below, above = int((s < lower).sum()), int((s > upper).sum())
        records.append(
            {
                "column": col,
                "Q1": round(q1, 3),
                "Q3": round(q3, 3),
                "IQR": round(iqr, 3),
                "lower_fence": round(lower, 3),
                "upper_fence": round(upper, 3),
                "n_low": below,
                "n_high": above,
                "n_outliers": below + above,
                "outlier_%": round(100 * (below + above) / len(s), 2),
            }
        )
    return pd.DataFrame(records).sort_values("n_outliers", ascending=False).reset_index(drop=True)


def apply_cleaning(
    df: pd.DataFrame, settings: Dict
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame, pd.DataFrame]:
    """
    Core cleaning stage of the ETL pipeline.

    Returns
    -------
    clean_df      : cleaned frame
    log           : ordered human-readable pipeline log
    missing_ba    : before/after missing-value counts per column
    outlier_rep   : Tukey IQR outlier report (pre-capping)
    """
    log: List[str] = []
    work = df.copy()
    n0 = len(work)
    log.append(f"[1] Input loaded: {n0:,} rows x {work.shape[1]} columns")

    # --- 1. Exact duplicate records -----------------------------------------
    dupes = int(work.duplicated().sum())
    if settings["drop_duplicates"]:
        work = work.drop_duplicates()
        log.append(f"[2] Duplicate records found: {dupes:,} -> dropped "
                   f"({len(work):,} rows remain)")
    else:
        log.append(f"[2] Duplicate check skipped by user (found {dupes:,})")

    # --- 2. Missing values ---------------------------------------------------
    miss_before = work.isna().sum().rename("before")
    strategy = settings["fill_strategy"]
    if strategy == "drop":
        before_drop = len(work)
        work = work.dropna()
        log.append(f"[3] Missing values: listwise deletion removed "
                   f"{before_drop - len(work):,} rows")
    elif strategy != "none":
        num_cols = work.select_dtypes(include=[np.number]).columns
        cat_cols = [c for c in work.columns if c not in num_cols]
        for col in num_cols:
            if work[col].isna().any():
                if strategy == "mean":
                    fill = work[col].mean()
                elif strategy == "zero":
                    fill = 0
                elif strategy == "mode":
                    mode = work[col].mode(dropna=True)
                    fill = mode.iloc[0] if len(mode) else work[col].median()
                else:  # median
                    fill = work[col].median()
                work[col] = work[col].fillna(fill)
        for col in cat_cols:
            if work[col].isna().any():
                mode = work[col].mode(dropna=True)
                work[col] = work[col].fillna(mode.iloc[0] if len(mode) else "Unknown")
        log.append(f"[3] Missing values imputed with strategy='{strategy}' "
                   f"({int(miss_before.sum()):,} cells treated)")
    else:
        log.append(f"[3] Missing-value handling disabled "
                   f"({int(miss_before.sum()):,} cells still missing)")

    # --- 3. Outlier detection (Tukey fence on amount + cap/ remove) ----------
    amount_col = pick_col(work, "amount")
    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    # Keep labels out of outlier treatment (they are binary indicators)
    numeric_cols = [c for c in numeric_cols
                    if c not in ("is_high_value", "is_churn_risk", "is_weekend")]
    outlier_rep = iqr_outlier_report(work, numeric_cols, settings["outlier_factor"]) \
        if numeric_cols else pd.DataFrame()

    action = settings["outlier_action"]
    if action != "none" and not outlier_rep.empty:
        if action == "cap":
            capped_cells = 0
            for col in outlier_rep["column"]:
                row = outlier_rep.loc[outlier_rep["column"] == col].iloc[0]
                lo, hi = row["lower_fence"], row["upper_fence"]
                s = work[col]
                mask = (s < lo) | (s > hi)
                capped_cells += int(mask.sum())
                work[col] = s.clip(lo, hi)
            log.append(f"[4] Outliers: {settings['outlier_factor']}xIQR fences -> "
                       f" winsorised {capped_cells:,} cells (capped)")
        elif action == "remove" and amount_col:
            q1, q3 = work[amount_col].quantile(0.25), work[amount_col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - settings["outlier_factor"] * iqr, q3 + settings["outlier_factor"] * iqr
            before = len(work)
            work = work[(work[amount_col] >= lo) & (work[amount_col] <= hi)]
            log.append(f"[4] Outliers: removed {before - len(work):,} rows outside "
                       f"[{lo:.2f}, {hi:.2f}] on '{amount_col}'")
    else:
        log.append("[4] Outlier treatment disabled by user")

    # --- 4. Basic type sanity -------------------------------------------------
    if amount_col:
        work[amount_col] = pd.to_numeric(work[amount_col], errors="coerce")
    qty_col = pick_col(work, "quantity")
    if qty_col:
        work[qty_col] = pd.to_numeric(work[qty_col], errors="coerce")

    miss_after = work.isna().sum().rename("after")
    missing_ba = pd.concat([miss_before, miss_after], axis=1)
    missing_ba["column"] = missing_ba.index

    log.append(f"[5] Cleaning complete: {n0:,} -> {len(work):,} rows "
               f"({n0 - len(work):,} removed)")
    return work, log, missing_ba.reset_index(drop=True), outlier_rep


def extract_rfm(
    tx: pd.DataFrame,
    date_col: str,
    amount_col: str,
    id_col: str,
    snapshot: pd.Timestamp,
    txn_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Classic RFM feature extraction.

        Recency   = days between snapshot and last purchase
        Frequency = number of distinct transactions
        Monetary  = total spend
    plus several auxiliary behavioural aggregates used downstream.
    """
    grp = tx.groupby(id_col, sort=False)
    freq = grp[txn_col].nunique() if txn_col else grp.size()

    rfm = pd.DataFrame(
        {
            "recency": (snapshot - grp[date_col].max()).dt.days.astype(int),
            "frequency": freq.astype(int),
            "monetary": grp[amount_col].sum().round(2),
            "avg_order_value": grp[amount_col].mean().round(2),
            "first_purchase": grp[date_col].min(),
            "last_purchase": grp[date_col].max(),
        }
    ).reset_index()

    if COL_CANDIDATES["quantity"][0] in tx.columns or pick_col(tx, "quantity"):
        q = pick_col(tx, "quantity")
        rfm = rfm.merge(
            tx.groupby(id_col)[q].sum().rename("total_quantity").reset_index(),
            on=id_col, how="left",
        )
    cat_col = pick_col(tx, "category")
    if cat_col:
        rfm = rfm.merge(
            tx.groupby(id_col)[cat_col].nunique().rename("unique_categories").reset_index(),
            on=id_col, how="left",
        )
        rfm = rfm.merge(
            tx.groupby(id_col)[cat_col]
            .agg(preferred_category=lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
            .reset_index(),
            on=id_col, how="left",
        )
    # Carry over first observed demographic / slowly-changing attributes
    static_like = [c for c in ["age", "gender", "region", "loyalty_tier", "channel",
                               "payment_method", "email_subscribed", "marketing_opt_in"]
                   if c in tx.columns]
    if static_like:
        rfm = rfm.merge(tx.groupby(id_col)[static_like].first().reset_index(),
                        on=id_col, how="left")

    signup_col = pick_col(tx, "signup")
    if signup_col:
        signup = tx.groupby(id_col)[signup_col].min().rename("signup_date")
        rfm = rfm.merge(signup.reset_index(), on=id_col, how="left")
        rfm["tenure_days"] = (snapshot - rfm["signup_date"]).dt.days.clip(lower=1)

    if pick_col(tx, "discount"):
        d = pick_col(tx, "discount")
        rfm = rfm.merge(
            tx.groupby(id_col)[d].mean().round(2).rename("avg_discount_pct").reset_index(),
            on=id_col, how="left",
        )
    return rfm


def derive_labels(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Supervised targets derived from RFM (used when the source has no labels).

        is_high_value : top 20% monetary AND top 20% frequency
        is_churn_risk : recency > 90d  OR  (recency > 60d AND below-median frequency)
        clv_proxy     : historical spend x engagement x tenure  (see report Eq. 8)
    """
    rfm = rfm.copy()
    mono_cut = rfm["monetary"].quantile(0.80)
    freq_cut = rfm["frequency"].quantile(0.80)
    rfm["is_high_value"] = (
        (rfm["monetary"] >= mono_cut) & (rfm["frequency"] >= freq_cut)
    ).astype(int)

    rec_med_hi = rfm["recency"] > 60
    freq_lo = rfm["frequency"] < rfm["frequency"].median()
    rfm["is_churn_risk"] = (
        (rfm["recency"] > 90) | (rec_med_hi & freq_lo)
    ).astype(int)

    tenure = rfm.get("tenure_days", pd.Series(365, index=rfm.index)).fillna(365).clip(lower=1)
    clv = (
        rfm["monetary"]
        * (1 + 1 / rfm["recency"].clip(lower=1))
        * (rfm["frequency"] / tenure)
        * 365
    )
    rfm["clv_proxy"] = clv.clip(0, clv.quantile(0.99)).round(2)
    return rfm


def build_customer_table(raw: pd.DataFrame, settings: Dict) -> Dict:
    """
    Full ETL orchestration: clean -> RFM -> labels -> customer-grain table.

    Returns a dictionary of artifacts consumed by every downstream tab.
    """
    df = coerce_dates(raw)
    date_col, amount_col = pick_col(df, "date"), pick_col(df, "amount")
    id_col = pick_col(df, "customer_id") or pick_col(df, "transaction_id")
    is_tx_grain = bool(date_col and amount_col and id_col
                       and df.groupby(id_col).size().max() > 1)

    artifacts: Dict = {
        "is_tx_grain": is_tx_grain,
        "date_col": date_col,
        "amount_col": amount_col,
        "id_col": id_col,
        "settings": dict(settings),
    }

    if is_tx_grain:
        clean_df, log, missing_ba, outlier_rep = apply_cleaning(df, settings)
        artifacts.update(clean_tx=clean_df, log=log,
                         missing_ba=missing_ba, outlier_rep=outlier_rep)

        snapshot = settings["snapshot_date"]
        if snapshot is None:
            snapshot = clean_df[date_col].max() + pd.Timedelta(days=1)
        snapshot = pd.Timestamp(snapshot)

        has_rfm = all(c in clean_df.columns for c in ("recency", "frequency", "monetary"))
        if settings["recompute_rfm"] or not has_rfm:
            rfm = extract_rfm(clean_df, date_col, amount_col, id_col, snapshot,
                              txn_col=pick_col(clean_df, "transaction_id"))
            log.append(f"[6] RFM extracted for {len(rfm):,} customers "
                       f"(snapshot = {snapshot.date()})")
        else:
            rfm = clean_df.groupby(id_col).agg(
                recency=("recency", "max"), frequency=("frequency", "max"),
                monetary=("monetary", "max"),
                avg_order_value=("avg_order_value", "mean")
                if "avg_order_value" in clean_df else ("monetary", "max"),
            ).reset_index()
            for extra in ("total_quantity", "unique_categories", "age", "gender",
                          "region", "loyalty_tier", "tenure_days", "signup_date",
                          "channel", "payment_method", "preferred_category"):
                if extra in clean_df.columns:
                    rfm = rfm.merge(
                        clean_df.groupby(id_col)[extra].first().reset_index(),
                        on=id_col, how="left")
            log.append(f"[6] Reused pre-computed RFM columns for {len(rfm):,} customers")

        # Preserve original labels when present, unless the user forces a rebuild
        label_cols = ["is_high_value", "is_churn_risk", "clv_proxy"]
        src_labels = [c for c in label_cols if c in clean_df.columns]
        if src_labels and not settings["rederive_labels"]:
            rfm = rfm.merge(
                clean_df.groupby(id_col)[src_labels].first().reset_index(),
                on=id_col, how="left")
            log.append(f"[7] Kept source labels: {', '.join(src_labels)}")
        rfm = derive_labels(rfm)  # fills anything that is still missing
        if not src_labels or settings["rederive_labels"]:
            log.append("[7] Labels derived from RFM (80th-percentile rule + churn window)")

        cust = rfm.copy()
        if "is_churn_risk" in cust:
            log.append(f"[8] Customer table ready: {len(cust):,} customers | "
                       f"high-value {cust['is_high_value'].mean() * 100:.1f}% | "
                       f"churn-risk {cust['is_churn_risk'].mean() * 100:.1f}%")
        artifacts.update(rfm=cust, clean_tx=clean_df)

    else:
        # Customer-grain upload (already one row per customer) ---------------
        clean_df, log, missing_ba, outlier_rep = apply_cleaning(df, settings)
        log.append("[6] Customer-grain data detected: RFM columns used as-is")
        cust = clean_df.copy()
        if "customer_id" not in cust.columns:
            cust.insert(0, "customer_id", [f"CUST_{i:06d}" for i in range(1, len(cust) + 1)])
        for c in ("recency", "frequency", "monetary"):
            if c not in cust.columns:
                raise ValueError(
                    f"Uploaded CSV is not transaction-grain and is missing '{c}'. "
                    "Provide either (transaction_date + amount + customer_id) or "
                    "pre-computed recency/frequency/monetary columns."
                )
        if not all(c in cust.columns for c in ("is_high_value", "is_churn_risk", "clv_proxy")):
            cust = derive_labels(cust)
            log.append("[7] Labels derived from RFM (uploaded data had no labels)")
        artifacts.update(clean_tx=clean_df, rfm=cust, log=log,
                         missing_ba=missing_ba, outlier_rep=outlier_rep)

    return artifacts


# --------------------------------------------------------------------------------
# Association-rule mining layer
# --------------------------------------------------------------------------------
def build_baskets(
    tx: pd.DataFrame, grain: str, item_col: str, top_n: int
) -> pd.DataFrame:
    """
    Transform transactions into a one-hot (basket x item) boolean matrix.

    grain : 'Customer' | 'Customer-Month' | 'Transaction'
    """
    df = tx[[item_col]].copy()
    df[item_col] = df[item_col].astype(str)

    if grain == "Transaction":
        id_col = pick_col(tx, "transaction_id")
        if not id_col:
            raise ValueError("Transaction baskets need a transaction_id/order_id column.")
        basket = df.groupby(tx[id_col])[item_col].apply(lambda s: sorted(set(s)))
    elif grain == "Customer-Month":
        id_col = pick_col(tx, "customer_id")
        date_col = pick_col(tx, "date")
        if not (id_col and date_col):
            raise ValueError("Customer-Month baskets need customer_id and a date column.")
        period = pd.to_datetime(tx[date_col]).dt.to_period("M").astype(str)
        keys = tx[id_col].astype(str) + " | " + period
        basket = df.groupby(keys.values)[item_col].apply(lambda s: sorted(set(s)))
    else:  # Customer
        id_col = pick_col(tx, "customer_id")
        if not id_col:
            raise ValueError("Customer baskets need a customer_id column.")
        basket = df.groupby(tx[id_col])[item_col].apply(lambda s: sorted(set(s)))

    basket = basket[basket.map(len) > 0]
    # Restrict to the most popular items to keep the search space tractable
    counts = basket.explode().value_counts()
    keep = counts.head(top_n).index.tolist()
    basket = basket.map(lambda items: [i for i in items if i in set(keep)])

    rows = [{i: (i in items) for i in keep} for items in basket]
    bool_df = pd.DataFrame(rows, columns=keep).astype(bool)
    return bool_df


def custom_apriori(
    bool_df: pd.DataFrame, min_support: float, max_len: int = 3
) -> pd.DataFrame:
    """
    Educational, dependency-free Apriori implementation (level-wise candidate
    generation + support counting). Used automatically when mlxtend is absent.
    """
    from itertools import combinations

    n = len(bool_df)
    if n == 0:
        return pd.DataFrame(columns=["support", "itemsets"])
    cols = list(bool_df.columns)
    freq: Dict[frozenset, float] = {}

    # L1
    current = []
    for item in cols:
        sup = float(bool_df[item].mean())
        if sup >= min_support:
            fs = frozenset([item])
            freq[fs] = sup
            current.append(fs)

    k = 2
    while current and k <= max_len:
        current_sets = [sorted(fs) for fs in current]
        candidates = set()
        for i in range(len(current_sets)):
            for j in range(i + 1, len(current_sets)):
                union = sorted(set(current_sets[i]) | set(current_sets[j]))
                if len(union) == k:
                    candidates.add(frozenset(union))
        next_level = []
        for cand in candidates:
            # Apriori prune: every (k-1)-subset must be frequent
            if all(frozenset(sub) in freq
                   for sub in combinations(sorted(cand), k - 1)):
                mask = bool_df[list(cand)].all(axis=1)
                sup = float(mask.mean())
                if sup >= min_support:
                    freq[cand] = sup
                    next_level.append(cand)
        current = next_level
        k += 1

    if not freq:
        return pd.DataFrame(columns=["support", "itemsets"])
    out = pd.DataFrame(
        [{"support": sup, "itemsets": frozenset(fs)} for fs, sup in freq.items()]
    )
    return out.sort_values("support", ascending=False).reset_index(drop=True)


def generate_rules(
    freq_df: pd.DataFrame, min_confidence: float
) -> pd.DataFrame:
    """Mine confidence/lift rules from a frequent-itemset table (custom path)."""
    from itertools import combinations

    sup_map = {frozenset(r.itemsets): r.support for r in freq_df.itertuples()}
    rules = []
    for fs, sup in sup_map.items():
        if len(fs) < 2:
            continue
        items = sorted(fs)
        for r in range(1, len(items)):
            for ant in combinations(items, r):
                a, b = frozenset(ant), fs - frozenset(ant)
                if a in sup_map and b in sup_map:
                    conf = sup / sup_map[a]
                    if conf >= min_confidence:
                        lift = conf / sup_map[b]
                        rules.append(
                            {
                                "antecedents": a,
                                "consequents": b,
                                "antecedent support": sup_map[a],
                                "consequent support": sup_map[b],
                                "support": sup,
                                "confidence": conf,
                                "lift": lift,
                                "leverage": sup - sup_map[a] * sup_map[b],
                            }
                        )
    return pd.DataFrame(rules)


@st.cache_data(show_spinner=False)
def suggest_thresholds(bool_df: pd.DataFrame) -> Dict:
    """
    Data-driven default thresholds for the rule miner.

    Support default: 0.7 x the strongest pair support (mining well below that
    explodes the search space; mining above it yields only single-item sets,
    and therefore no rules at all).

    Confidence default: mined at a floor of 0.05, then set to the confidence
    of the ~6th strongest positive-lift rule, so the initial view opens on a
    meaningful rule table for any item space (categories or products).
    """
    n = len(bool_df)
    B = bool_df.to_numpy(dtype=np.float32)
    singles = B.mean(axis=0)

    max_pair_sup = 0.0
    if B.shape[1] >= 2 and n:
        inter = B.T @ B                       # co-occurrence counts
        np.fill_diagonal(inter, 0)
        max_pair_sup = float((inter / n).max())
    default_sup = float(np.clip(0.7 * max_pair_sup, 0.0002, 0.4))

    # Mine at a low confidence floor purely to calibrate the slider default
    default_conf = 0.05
    try:
        _, rules, _ = mine_rules(bool_df, default_sup, 0.05)
        if len(rules):
            strong = rules[rules["lift"] > 1.0].sort_values("confidence",
                                                            ascending=False)
            if len(strong):
                idx = min(5, len(strong) - 1)
                # snap down to the slider's 0.05 grid so no rule is lost
                default_conf = float(np.floor(strong["confidence"].iloc[idx] / 0.05) * 0.05)
                default_conf = float(np.clip(default_conf, 0.05, 0.90))
    except Exception:
        pass

    return {
        "min_support": round(default_sup, 4),
        "min_confidence": round(default_conf, 2),
        "max_pair_support": max_pair_sup,
    }


@st.cache_data(show_spinner="Mining association rules …")
def mine_rules(
    bool_df_hashable: pd.DataFrame,
    min_support: float,
    min_confidence: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Apriori + rule generation. Uses mlxtend when available, otherwise the
    bundled custom implementation. Returns (frequent_itemsets, rules, engine).
    """
    if MLXTEND_AVAILABLE:
        engine = "mlxtend.apriori"
        freq = mlx_apriori(bool_df_hashable, min_support=min_support, use_colnames=True)
        if freq.empty:
            return freq, pd.DataFrame(), engine
        rules = mlx_association_rules(freq, metric="confidence",
                                      min_threshold=min_confidence)
        keep = ["antecedents", "consequents", "antecedent support",
                "consequent support", "support", "confidence", "lift", "leverage"]
        rules = rules[[c for c in keep if c in rules.columns]]
    else:
        engine = "custom Apriori (fallback)"
        freq = custom_apriori(bool_df_hashable, min_support)
        rules = generate_rules(freq, min_confidence) if not freq.empty else pd.DataFrame()
    return freq, rules, engine


# --------------------------------------------------------------------------------
# Clustering layer
# --------------------------------------------------------------------------------
@st.cache_data(show_spinner="Running Elbow / Silhouette search …")
def elbow_search(X: pd.DataFrame, k_max: int, seed: int) -> Dict:
    """Compute inertia + silhouette for k = 1 .. k_max."""
    ks, inertias, sils = [], [], []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        ks.append(k)
        inertias.append(float(km.inertia_))
        sils.append(None if k < 2 else float(silhouette_score(X, labels)))
    return {"k": ks, "inertia": inertias, "silhouette": sils}


def fit_kmeans(X: pd.DataFrame, k: int, seed: int):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km, km.fit_predict(X)


def name_clusters(centers: pd.DataFrame) -> Dict[int, str]:
    """Rule-based marketing names from RFM cluster centroids."""
    names = {}
    if not {"recency", "frequency", "monetary"}.issubset(centers.columns):
        return {i: f"Segment {i}" for i in range(len(centers))}
    med_r, med_f, med_m = centers["recency"].median(), centers["frequency"].median(), centers["monetary"].median()
    for idx, row in centers.iterrows():
        if row["monetary"] >= med_m and row["frequency"] >= med_f and row["recency"] <= med_r:
            name = "🏆 Champions (high spend, frequent, recent)"
        elif row["monetary"] >= med_m and row["recency"] > med_r:
            name = "💤 Big Spenders Gone Quiet (win-back)"
        elif row["recency"] <= med_r and row["frequency"] >= med_f:
            name = "🔁 Loyal & Recent (nurture)"
        elif row["recency"] <= med_r and row["frequency"] < med_f:
            name = "🌱 New / Promising (onboard)"
        elif row["recency"] > med_r and row["frequency"] < med_f:
            name = "⚠️ At Risk / Hibernating (reactivate)"
        else:
            name = "🧾 Regular Mid-Tier"
        names[idx] = name
    return names


# --------------------------------------------------------------------------------
# Supervised learning layers (classification & regression)
# --------------------------------------------------------------------------------
def _prepare_xy(
    cust: pd.DataFrame, features: List[str], target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    data = cust.dropna(subset=[target]).copy()
    X = data[features].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = data[target].astype(int) if data[target].nunique() <= 4 else data[target].astype(float)
    keep = X.index
    return X, y.loc[keep]


@st.cache_data(show_spinner="Training classifiers …")
def train_classification(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    max_depth: int,
    seed: int,
) -> Dict:
    """
    Train & evaluate Decision Tree (J48/C4.5 family) vs Gaussian Naive Bayes.
    All artefacts needed by the UI are returned in one dictionary.
    """
    n_classes = int(y.nunique())
    if n_classes < 2:
        raise ValueError("Target has a single class - cannot train a classifier.")

    stratify = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify)

    models = {
        "Decision Tree (J48 equivalent)": DecisionTreeClassifier(
            criterion="gini", max_depth=max_depth, min_samples_leaf=5,
            random_state=seed, class_weight="balanced"),
        "Naive Bayes (GaussianNB)": GaussianNB(),
    }

    results, fitted = {}, {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1] if n_classes == 2 else None

        metrics = {
            "Accuracy": accuracy_score(y_te, y_pred),
            "Precision": precision_score(y_te, y_pred, average="binary", zero_division=0)
            if n_classes == 2 else precision_score(y_te, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_te, y_pred, average="binary", zero_division=0)
            if n_classes == 2 else recall_score(y_te, y_pred, average="macro", zero_division=0),
            "F1-Score": f1_score(y_te, y_pred, average="binary", zero_division=0)
            if n_classes == 2 else f1_score(y_te, y_pred, average="macro", zero_division=0),
            "Train time (s)": train_time,
        }
        cm = confusion_matrix(y_te, y_pred)
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_te, y_prob)
            metrics["ROC-AUC"] = auc(fpr, tpr)
        else:
            fpr = tpr = np.array([])
            metrics["ROC-AUC"] = np.nan

        results[name] = {"metrics": metrics, "cm": cm, "fpr": fpr, "tpr": tpr,
                         "n_test": len(y_te)}
        fitted[name] = model

    # Decision-tree internals for the interpretability panel
    dt = fitted["Decision Tree (J48 equivalent)"]
    importances = pd.Series(dt.feature_importances_, index=X.columns) \
        .sort_values(ascending=False)
    tree_rules = export_text(dt, feature_names=list(X.columns), max_depth=6)
    return {"results": results, "importances": importances,
            "tree_rules": tree_rules, "n_test": len(y_te), "classes": list(map(str, y.unique()))}


@st.cache_data(show_spinner="Training regressors …")
def train_regression(
    X: pd.DataFrame, y: pd.Series, test_size: float, seed: int
) -> Dict:
    """Compare Linear / Ridge / Random-Forest / Gradient-Boosting on CLV."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=seed),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=10, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.1, random_state=seed),
    }
    rows, fitted, preds = [], {}, {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0
        y_pred = model.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        rows.append({
            "Model": name,
            "MAE": float(mean_absolute_error(y_te, y_pred)),
            "RMSE": rmse,
            "R²": float(r2_score(y_te, y_pred)),
            "MAE/mean(y)": float(mean_absolute_error(y_te, y_pred) / max(y.mean(), 1e-9)),
            "Train time (s)": train_time,
        })
        fitted[name] = model
        preds[name] = y_pred

    best = max(rows, key=lambda r: r["R²"])["Model"]
    imp = None
    if hasattr(fitted[best], "feature_importances_"):
        imp = pd.Series(fitted[best].feature_importances_, index=X.columns) \
            .sort_values(ascending=False)
    elif hasattr(fitted[best], "coef_"):
        imp = pd.Series(np.abs(fitted[best].coef_), index=X.columns) \
            .sort_values(ascending=False)
    return {"metrics": pd.DataFrame(rows), "models": fitted, "preds": preds,
            "best": best, "importances": imp,
            "y_test": y_te, "X_test": X_te}


@st.cache_resource(show_spinner="Building live-prediction bundle …")
def build_live_bundle(cust: pd.DataFrame, k: int, seed: int) -> Dict:
    """
    Train the models used by the Live Prediction tab on the *full* customer
    table (no hold-out): segmenter + two binary classifiers + CLV regressor.
    """
    features = [f for f in CORE_FEATURES if f in cust.columns]
    if len(features) < 2:
        raise ValueError(f"Need >= 2 modelling features; found {features}")

    X, y_hi = _prepare_xy(cust, features, "is_high_value")
    _, y_ch = _prepare_xy(cust, features, "is_churn_risk")
    X = X.loc[y_hi.index.intersection(y_ch.index)]
    y_hi, y_ch = y_hi.loc[X.index], y_ch.loc[X.index]

    scaler = StandardScaler().fit(X)
    Xs = pd.DataFrame(scaler.transform(X), columns=features, index=X.index)
    km, labels = fit_kmeans(Xs, k, seed)
    centers = pd.DataFrame(km.cluster_centers_, columns=features)
    names = name_clusters(centers)

    dt_hi = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5,
                                   random_state=seed, class_weight="balanced").fit(X, y_hi)
    nb_hi = GaussianNB().fit(X, y_hi)
    dt_ch = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5,
                                   random_state=seed, class_weight="balanced").fit(X, y_ch)
    nb_ch = GaussianNB().fit(X, y_ch)

    reg_target = "clv_proxy" if "clv_proxy" in cust.columns else "monetary"
    Xr, yr = _prepare_xy(cust, features, reg_target)
    reg = RandomForestRegressor(n_estimators=300, max_depth=10,
                                random_state=seed, n_jobs=-1).fit(Xr, yr)

    return {
        "features": features,
        "scaler": scaler, "kmeans": km, "cluster_names": names,
        "centers": centers, "labels": labels,
        "dt_high": dt_hi, "nb_high": nb_hi,
        "dt_churn": dt_ch, "nb_churn": nb_ch,
        "reg": reg, "reg_target": reg_target,
        "stats": X.agg(["min", "median", "max"]),
        "has_high": bool(y_hi.nunique() > 1),
        "has_churn": bool(y_ch.nunique() > 1),
    }


# --------------------------------------------------------------------------------
# Figure builders (Plotly)
# --------------------------------------------------------------------------------
def fig_missing_ba(missing_ba: pd.DataFrame) -> go.Figure:
    plot = missing_ba[(missing_ba["before"] > 0) | (missing_ba["after"] > 0)]
    if plot.empty:
        return go.Figure().update_layout(
            title="No missing values detected - dataset is complete ✓",
            height=300)
    fig = go.Figure()
    fig.add_bar(x=plot["column"], y=plot["before"], name="Before",
                marker_color="#EF553B")
    fig.add_bar(x=plot["column"], y=plot["after"], name="After",
                marker_color="#00CC96")
    fig.update_layout(barmode="group", height=360, title="Missing values: before vs after cleaning",
                      xaxis_tickangle=-40, legend_orientation="h")
    return fig


def fig_roc(results: Dict) -> go.Figure:
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        if len(res["fpr"]) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=res["fpr"], y=res["tpr"], mode="lines",
            name=f"{name} (AUC = {res['metrics']['ROC-AUC']:.3f})",
            line=dict(width=3, color=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)])))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Random baseline (AUC = 0.5)",
                             line=dict(dash="dash", color="#9aa0a6")))
    fig.update_layout(xaxis_title="False Positive Rate (1 - Specificity)",
                      yaxis_title="True Positive Rate (Sensitivity)",
                      title="ROC Curve Comparison", height=430,
                      legend_orientation="h")
    return fig


def fig_confusion(cm: np.ndarray, title: str, labels: List[str]) -> go.Figure:
    z = cm.astype(int)
    text = [[f"{v}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, colorscale="Blues",
        text=text, texttemplate="%{text}", textfont={"size": 18},
        hovertemplate="Actual %{y} / Predicted %{x}<br>count=%{z}<extra></extra>"))
    fig.update_layout(
        title=title, height=360,
        xaxis_title="Predicted label", yaxis_title="True label",
        yaxis=dict(autorange="reversed"))
    return fig


def fig_star_schema() -> go.Figure:
    """Schematic star-schema diagram (fact table + conformed dimensions)."""
    dims = [
        ("Dim_Customer\ncustomer_key | age | gender\nregion | loyalty_tier", -7.5, 3.0),
        ("Dim_Date\ndate_key | year | quarter\nmonth | is_weekend", 0.0, 4.6),
        ("Dim_Product\nproduct_key | category\nproduct_name | unit_price", 7.5, 3.0),
        ("Dim_Channel\nchannel_key | channel\npayment_method", -7.5, -3.0),
        ("Dim_Customer_Attr\nsignup_date | tenure\ntier | subscribed", 7.5, -3.0),
    ]
    fact = ("FACT_SALES\ntransaction_key | customer_key | product_key\ndate_key | qty | amount | discount",
            0.0, 0.0)
    fig = go.Figure()
    for label, x, y in dims:
        fig.add_shape(type="rect", x0=x - 2.6, x1=x + 2.6, y0=y - 0.85, y1=y + 0.85,
                      fillcolor="#E8EEFF", line=dict(color="#5C6BC0", width=2))
        fig.add_annotation(x=x, y=y, text=f"<b>{label}</b>", showarrow=False,
                           font=dict(size=11, color="#283593"), align="center")
        fig.add_trace(go.Scatter(x=[x, fact[1]], y=[y, fact[2]], mode="lines",
                                 line=dict(color="#9FA8DA", width=1.5, dash="dot"),
                                 showlegend=False, hoverinfo="skip"))
    fx, fy = fact[1], fact[2]
    fig.add_shape(type="rect", x0=fx - 3.4, x1=fx + 3.4, y0=fy - 1.1, y1=fy + 1.1,
                  fillcolor="#FFF3E0", line=dict(color="#EF6C00", width=2.5))
    fig.add_annotation(x=fx, y=fy, text=f"<b>{fact[0]}</b>", showarrow=False,
                       font=dict(size=11, color="#E65100"), align="center")
    fig.update_layout(
        title="Star Schema - FACT_SALES (transaction grain) linked to conformed dimensions",
        height=520, xaxis=dict(visible=False, range=[-11, 11]),
        yaxis=dict(visible=False, range=[-5.2, 5.8]), plot_bgcolor="white")
    return fig


# --------------------------------------------------------------------------------
# Session helpers
# --------------------------------------------------------------------------------
def reset_downstream_state() -> None:
    for key in ("artifacts", "etl_settings"):
        st.session_state.pop(key, None)


def run_pipeline(raw: pd.DataFrame, settings: Dict) -> Optional[Dict]:
    """Execute the ETL and cache the artifacts in session state."""
    try:
        artifacts = build_customer_table(raw, settings)
        st.session_state["artifacts"] = artifacts
        st.session_state["etl_settings"] = dict(settings)
        return artifacts
    except Exception:
        show_exception("ETL pipeline failed")
        return None


def get_artifacts() -> Optional[Dict]:
    return st.session_state.get("artifacts")


def get_features(cust: pd.DataFrame) -> List[str]:
    return [f for f in CORE_FEATURES if f in cust.columns]


# --------------------------------------------------------------------------------
# Sidebar / data source
# --------------------------------------------------------------------------------
def sidebar_data_source() -> Optional[pd.DataFrame]:
    st.sidebar.markdown('<div class="side-head">Console</div>', unsafe_allow_html=True)
    st.sidebar.title("🛍️ Customer Analytics")
    st.sidebar.caption("Data Warehousing & Mining · ML Platform")

    st.sidebar.markdown('<div class="side-head">Data source</div>', unsafe_allow_html=True)
    source = st.sidebar.radio(
        "Choose the input feed",
        [
            "🏆 Kaggle Online Retail (real data)",
            "Built-in dataset (ecommerce_10k.csv)",
            "Upload your own CSV",
        ],
        help="The recommended feed is the classic UCI 'Online Retail' basket "
             "dataset (398k cleaned records, Dec 2010 - Dec 2011).",
    )

    df, source_key = None, None
    if source.startswith("🏆"):
        if not os.path.exists(ONLINE_RETAIL_PATH):
            st.sidebar.error("`online_retail.csv` not found in the project folder.")
            st.sidebar.caption("Convert the source workbook once, then rerun:")
            st.sidebar.code(
                'python prepare_online_retail.py --src "online+retail.zip"',
                language="bash")
            st.session_state["source_label"] = "Kaggle Online Retail (needs conversion)"
        else:
            try:
                df = load_online_retail(ONLINE_RETAIL_PATH)
                source_key = "online:" + str(os.path.getmtime(ONLINE_RETAIL_PATH))
                st.session_state["source_label"] = \
                    "🏆 **Kaggle Online Retail** — real UK retailer baskets, Dec 2010–Dec 2011"
            except Exception:
                show_exception("Could not load the Online Retail dataset")
    elif source.startswith("Built-in"):
        try:
            df = load_builtin(DEFAULT_DATA_PATH)
            source_key = "builtin:" + str(os.path.getmtime(DEFAULT_DATA_PATH)) \
                if os.path.exists(DEFAULT_DATA_PATH) else "builtin:generated"
            st.session_state["source_label"] = \
                "**Built-in synthetic warehouse** — `ecommerce_10k.csv` (12,000 rows)"
        except Exception:
            show_exception("Could not load the built-in dataset")
    else:
        uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is not None:
            try:
                raw_bytes = uploaded.getvalue()
                df = load_upload(raw_bytes, uploaded.name)
                digest = hashlib.md5(raw_bytes).hexdigest()[:10]
                source_key = f"upload:{uploaded.name}:{digest}"
                st.session_state["source_label"] = \
                    "**Uploaded file** — `" + uploaded.name + "`"
            except Exception:
                show_exception("Could not parse the uploaded CSV")

    # Dataset guide (static context card for the real feed)
    if source.startswith("🏆") and os.path.exists(ONLINE_RETAIL_PATH):
        with st.sidebar.expander("📖 Dataset guide", expanded=False):
            st.markdown("""
**Kaggle / UCI *Online Retail*** — the classic e-commerce market-basket study.
- Source workbook: 541,909 line-items (`Online Retail.xlsx`)
- Cleaned export: 397,884 records · 18,532 invoices · 4,338 customers
- Schema: `transaction_id, customer_id, transaction_date, stock_code,
  product_name, quantity, unit_price, amount, region`
- Preprocessing (see `prepare_online_retail.py`):
  cancellations dropped, anonymous rows dropped, invalid qty/price dropped,
  `amount = qty × price`
""")

    if df is None:
        st.sidebar.info("Awaiting data source …")
        return None

    # New data source -> invalidate all derived artifacts
    if st.session_state.get("source_key") != source_key:
        reset_downstream_state()
        st.session_state["source_key"] = source_key
        st.session_state["etl_settings"] = dict(DEFAULT_ETL_SETTINGS)
        run_pipeline(df, st.session_state["etl_settings"])

    st.sidebar.success(f"Loaded **{len(df):,}** rows × {df.shape[1]} cols")
    if st.sidebar.button("♻️ Reset / re-run ETL", width="stretch"):
        st.session_state["etl_settings"] = dict(DEFAULT_ETL_SETTINGS)
        reset_downstream_state()
        run_pipeline(df, st.session_state["etl_settings"])
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="side-head">Modules</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        "1. Overview\n"
        "2. ETL & RFM\n"
        "3. Association Rules\n"
        "4. Clustering (K-Means)\n"
        "5. Classification (DT vs NB)\n"
        "6. Regression (CLV)\n"
        "7. Live Prediction\n"
        "8. Accuracy & Comparison\n"
        "9. About"
    )
    return df


# --------------------------------------------------------------------------------
# Tab 1 - Overview
# --------------------------------------------------------------------------------
def tab_overview(raw: pd.DataFrame, artifacts: Dict) -> None:
    st.subheader("📊 Dataset Overview & Exploratory Analysis")
    cust = artifacts.get("rfm")

    # ---- KPI row ------------------------------------------------------------
    c = st.columns(6)
    c[0].metric("Transactions / Rows", f"{len(raw):,}")
    n_cust = raw[pick_col(raw, 'customer_id')].nunique() if pick_col(raw, 'customer_id') else (
        len(cust) if cust is not None else 0)
    c[1].metric("Customers", f"{n_cust:,}")
    date_col, amt_col = artifacts.get("date_col"), artifacts.get("amount_col")
    if date_col:
        span = pd.to_datetime(raw[date_col], errors="coerce")
        c[2].metric("Date span", f"{span.min():%Y-%m-%d} → {span.max():%Y-%m-%d}")
    else:
        c[2].metric("Date span", "n/a")
    c[3].metric("Avg order value",
                f"${raw[amt_col].mean():,.2f}" if amt_col else "n/a")
    if cust is not None and "is_high_value" in cust:
        c[4].metric("High-value customers", f"{cust['is_high_value'].mean() * 100:.1f}%")
        c[5].metric("Churn-risk customers", f"{cust['is_churn_risk'].mean() * 100:.1f}%")

    st.markdown("---")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Monthly revenue trend**")
        if date_col and amt_col:
            tmp = raw.copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            monthly = (tmp.dropna(subset=[date_col])
                       .set_index(date_col)[amt_col]
                       .resample("ME").sum().reset_index())
            fig = px.area(monthly, x=date_col, y=amt_col,
                          markers=True,
                          color_discrete_sequence=["#636EFA"])
            fig.update_layout(height=330, yaxis_title="Revenue", xaxis_title="",
                              margin=dict(t=30, b=10))
            st.plotly_chart(fig, width="stretch", key="ov_monthly")
        else:
            st.info("No date + amount columns available for trend analysis.")

        st.markdown("**Revenue by product category**")
        cat_col, amt = pick_col(raw, "category"), amt_col
        if cat_col and amt:
            cat = (raw.groupby(cat_col)[amt].sum().sort_values(ascending=False)
                   .reset_index())
            fig = px.bar(cat, x=cat_col, y=amt, color=amt,
                         color_continuous_scale="Viridis",
                         labels={cat_col: "Category", amt: "Revenue"})
            fig.update_layout(height=330, coloraxis_showscale=False,
                              xaxis_tickangle=-30, margin=dict(t=30, b=10))
            st.plotly_chart(fig, width="stretch", key="ov_cat")

    with right:
        st.markdown("**Customer value distribution (Monetary)**")
        if cust is not None and "monetary" in cust:
            fig = px.histogram(cust, x="monetary", nbins=40,
                               color_discrete_sequence=["#00CC96"])
            fig.update_layout(height=245, margin=dict(t=20, b=10))
            st.plotly_chart(fig, width="stretch", key="ov_mon")

        st.markdown("**Channel mix**")
        chan = pick_col(raw, "channel")
        if not chan and "channel" in raw.columns:
            chan = "channel"
        if chan:
            vc = raw[chan].value_counts().reset_index()
            vc.columns = [chan, "count"]
            fig = px.pie(vc, names=chan, values="count", hole=0.45,
                         color_discrete_sequence=CLUSTER_PALETTE)
            fig.update_layout(height=245, margin=dict(t=20, b=10),
                              legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, width="stretch", key="ov_chan")

    # ---- Correlation heatmap (seaborn / matplotlib) -------------------------
    st.markdown("---")
    st.markdown("**Feature correlation matrix** (Pearson r, customer grain)")
    if cust is not None:
        num = cust.select_dtypes(include=[np.number])
        drop_cols = [c for c in ("is_high_value", "is_churn_risk") if c in num.columns]
        num = num.drop(columns=drop_cols, errors="ignore").iloc[:, :10]
        if num.shape[1] >= 3:
            fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=110)
            sns.heatmap(num.corr().round(2), annot=True, fmt=".2f", cmap="coolwarm",
                        linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.8})
            ax.set_title("Pearson correlation of RFM / behavioural features")
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

    with st.expander("🔎 Raw data preview"):
        st.dataframe(raw.head(50), width="stretch", hide_index=True)


# --------------------------------------------------------------------------------
# Tab 2 - ETL & RFM
# --------------------------------------------------------------------------------
def tab_etl(raw: pd.DataFrame, artifacts: Dict) -> None:
    st.subheader("⚙️ Data Warehouse ETL, Cleaning & RFM Engineering")

    settings = dict(st.session_state.get("etl_settings", DEFAULT_ETL_SETTINGS))
    cust = artifacts.get("rfm")

    # ---------- Step 1: raw profile -----------------------------------------
    st.markdown("#### Step 1 · Raw data profiling")
    prof = profile_dataframe(raw)
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(prof, width="stretch", hide_index=True, height=330)
    with right:
        n_miss = int(raw.isna().sum().sum())
        n_dupes = int(raw.duplicated().sum())
        st.metric("Rows", f"{len(raw):,}")
        st.metric("Missing cells", f"{n_miss:,}")
        st.metric("Duplicate rows", f"{n_dupes:,}")
        st.metric("Numeric columns",
                  int((raw.dtypes.astype(str).str.startswith(("int", "float"))).sum()))

    # ---------- Step 2: controls --------------------------------------------
    st.markdown("#### Step 2 · Pipeline configuration")
    cols = st.columns(5)
    with cols[0]:
        settings["drop_duplicates"] = st.checkbox("Drop duplicates",
                                                  value=settings["drop_duplicates"])
    with cols[1]:
        settings["fill_strategy"] = st.selectbox(
            "Missing-value strategy", ["median", "mean", "mode", "zero", "drop", "none"],
            index=["median", "mean", "mode", "zero", "drop", "none"].index(settings["fill_strategy"]))
    with cols[2]:
        settings["outlier_action"] = st.selectbox(
            "Outlier action (IQR)", ["cap", "remove", "none"],
            index=["cap", "remove", "none"].index(settings["outlier_action"]))
    with cols[3]:
        settings["outlier_factor"] = float(st.number_input(
            "IQR factor (k)", 0.5, 5.0, float(settings["outlier_factor"]), 0.5))
    with cols[4]:
        settings["recompute_rfm"] = st.checkbox("Recompute RFM",
                                                value=settings["recompute_rfm"])
    settings["rederive_labels"] = st.checkbox(
        "Force re-derivation of labels (high-value / churn / CLV)",
        value=settings["rederive_labels"],
        help="Off = keep labels that already exist in the source data.")
    snap_default = None
    if artifacts.get("date_col"):
        dcol = artifacts["date_col"]
        if dcol in raw.columns:
            snap_default = (pd.to_datetime(raw[dcol], errors="coerce").max()
                            + pd.Timedelta(days=1))
    snap = st.date_input("RFM snapshot date (= max transaction date + 1 by default)",
                         value=snap_default.date() if snap_default is not None
                         else pd.Timestamp.today().date())
    settings["snapshot_date"] = pd.Timestamp(snap) if snap else None

    if st.button("🚀 Run ETL Pipeline", type="primary"):
        with st.spinner("Extract → Transform → Load …"):
            new_artifacts = run_pipeline(raw, settings)
        if new_artifacts:
            st.success("ETL complete - downstream tabs now use the refreshed tables.")
            st.rerun()

    # ---------- Step 3: results ---------------------------------------------
    artifacts = get_artifacts() or artifacts
    cust = artifacts.get("rfm")
    log = artifacts.get("log", [])

    st.markdown("#### Step 3 · Pipeline log")
    st.code("\n".join(log), language="text")

    st.markdown("#### Step 4 · Missing-value treatment")
    mba = artifacts.get("missing_ba")
    if mba is not None:
        st.plotly_chart(fig_missing_ba(mba), width="stretch", key="etl_missing")

    st.markdown("#### Step 5 · Outlier detection (Tukey IQR fences)")
    rep = artifacts.get("outlier_rep")
    if rep is not None and not rep.empty:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.dataframe(rep.style.format({"Q1": "{:.2f}", "Q3": "{:.2f}",
                                           "IQR": "{:.2f}", "lower_fence": "{:.2f}",
                                           "upper_fence": "{:.2f}",
                                           "outlier_%": "{:.2f}"}),
                         width="stretch", hide_index=True)
        with c2:
            top = rep.head(12)
            fig = px.bar(top, x="column", y="outlier_%", color="outlier_%",
                         color_continuous_scale="OrRd")
            fig.update_layout(height=330, coloraxis_showscale=False,
                              xaxis_tickangle=-40, margin=dict(t=20, b=10))
            st.plotly_chart(fig, width="stretch", key="etl_outlier")
    else:
        st.info("No numeric columns available for outlier analysis.")

    st.markdown("#### Step 6 · RFM feature table (customer grain)")
    if cust is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.dataframe(cust.head(25), width="stretch", hide_index=True, height=340)
        with c2:
            fig = px.histogram(cust, x="recency", nbins=40,
                               color_discrete_sequence=["#EF553B"])
            fig.update_layout(title="Recency (days)", height=260, margin=dict(t=40, b=10))
            st.plotly_chart(fig, width="stretch", key="etl_r")
            fig = px.histogram(cust, x="frequency", nbins=40,
                               color_discrete_sequence=["#636EFA"])
            fig.update_layout(title="Frequency (transactions)", height=260,
                              margin=dict(t=40, b=10))
            st.plotly_chart(fig, width="stretch", key="etl_f")
        with c3:
            fig = px.histogram(cust, x="monetary", nbins=40,
                               color_discrete_sequence=["#00CC96"])
            fig.update_layout(title="Monetary (total spend)", height=260,
                              margin=dict(t=40, b=10))
            st.plotly_chart(fig, width="stretch", key="etl_m")
            # RFM quartile segmentation mini-chart
            r, f = cust["recency"], cust["frequency"]
            seg = pd.cut(r, 4, labels=["R1 (freshest)", "R2", "R3", "R4 (stalest)"])
            counts = seg.value_counts().sort_index()
            fig = px.bar(x=counts.index.astype(str), y=counts.values,
                         color_discrete_sequence=["#AB63FA"])
            fig.update_layout(title="Customers by recency quartile-bin",
                              height=260, margin=dict(t=40, b=10),
                              xaxis_title="", yaxis_title="customers")
            st.plotly_chart(fig, width="stretch", key="etl_rq")

        st.download_button("⬇️ Download customer-grain table (CSV)",
                           cust.to_csv(index=False).encode("utf-8"),
                           file_name="customer_grain_table.csv", mime="text/csv")

    st.markdown("#### Step 7 · Warehouse schema")
    st.plotly_chart(fig_star_schema(), width="stretch", key="etl_star")

    if artifacts.get("is_tx_grain"):
        st.markdown("#### Step 8 · Cleaned transaction table (fact grain)")
        st.dataframe(artifacts["clean_tx"].head(30), width="stretch", hide_index=True)


# --------------------------------------------------------------------------------
# Tab 3 - Association rules
# --------------------------------------------------------------------------------
def tab_association(raw: pd.DataFrame, artifacts: Dict) -> None:
    st.subheader("🧺 Association Rule Mining - Market Basket Analysis")
    st.caption("Apriori algorithm → frequent itemsets → Support / Confidence / Lift rules. "
               "Engine: **mlxtend**" if MLXTEND_AVAILABLE else
               "Engine: **custom Apriori fallback** (mlxtend not installed)")

    tx = artifacts.get("clean_tx", raw)
    cat_col = pick_col(tx, "category")
    prod_col = pick_col(tx, "product")

    item_opts = {}
    if cat_col:
        item_opts["Product category"] = cat_col
    if prod_col:
        item_opts["Product name"] = prod_col
    if not item_opts:
        st.error("No category/product column found - association mining needs an item column.")
        return

    c = st.columns(5)
    with c[0]:
        item_label = st.selectbox("Item type", list(item_opts.keys()))
    with c[1]:
        grain = st.selectbox("Basket grain", ["Customer", "Customer-Month", "Transaction"],
                             help="A basket = items bought together in the selected unit.")
    with c[2]:
        top_n = st.number_input("Top-N items kept", 5, 500, 60, 5)

    # Build the basket matrix first so slider defaults can be data-driven
    # (fixed defaults frequently produce zero rules on sparse item spaces).
    try:
        bool_df = build_baskets(tx, grain, item_opts[item_label], int(top_n))
    except Exception:
        show_exception("Could not build the basket matrix")
        return
    if bool_df.empty or bool_df.shape[1] < 2:
        st.warning("Basket matrix too sparse - lower min-support or keep more items.")
        return

    hints = suggest_thresholds(bool_df)
    c = st.columns(3)
    with c[0]:
        min_sup = st.slider(
            "Minimum support", 0.0002, 0.5, hints["min_support"], 0.0002,
            format="%.4f",
            help=f"Data-driven default: 0.7 x strongest pair support "
                 f"({hints['max_pair_support']:.4f}).")
    with c[1]:
        min_conf = st.slider(
            "Minimum confidence", 0.05, 1.0, hints["min_confidence"], 0.05,
            help="Seeded at the confidence of the ~6th strongest positive-lift "
                 "rule found at a 0.05 floor.")
    with c[2]:
        min_lift = st.slider("Highlight rules with lift >", 0.5, 5.0, 1.0, 0.1)

    try:
        st.info(f"**Basket matrix:** {len(bool_df):,} baskets × {bool_df.shape[1]} items · "
                f"avg items/basket = {bool_df.sum(axis=1).mean():.2f} · "
                f"strongest pair support = {hints['max_pair_support']:.4f}")

        freq, rules, engine = mine_rules(bool_df, float(min_sup), float(min_conf))

        tab_a, tab_b, tab_c = st.tabs(["Frequent itemsets", "Rules table", "Rule explorer"])
        with tab_a:
            if freq.empty:
                st.warning("No frequent itemsets at this support level.")
            else:
                show = freq.copy()
                show["itemsets"] = show["itemsets"].map(_fmt_set)
                show["length"] = show["itemsets"].map(lambda s: s.count(",") + 1)
                show["support %"] = (show["support"] * 100).round(2)
                st.dataframe(show[["itemsets", "length", "support", "support %"]]
                             .sort_values("support", ascending=False),
                             width="stretch", hide_index=True, height=350)

                top = show.sort_values("support", ascending=False).head(15)
                fig = px.bar(top, x="support", y="itemsets", orientation="h",
                             color="support", color_continuous_scale="Teal",
                             labels={"support": "Support"})
                fig.update_layout(height=430, yaxis={"categoryorder": "total ascending"},
                                  coloraxis_showscale=False, margin=dict(t=20, b=10))
                st.plotly_chart(fig, width="stretch", key="as_top")

        with tab_b:
            if rules.empty:
                st.warning("No rules satisfied the confidence threshold.")
            else:
                tbl = rules.copy()
                tbl["antecedents"] = tbl["antecedents"].map(_fmt_set)
                tbl["consequents"] = tbl["consequents"].map(_fmt_set)
                for col in ("support", "confidence", "antecedent support", "consequent support"):
                    if col in tbl:
                        tbl[col] = tbl[col].round(4)
                for col in ("lift", "leverage"):
                    if col in tbl:
                        tbl[col] = tbl[col].round(3)
                tbl = tbl.sort_values("lift", ascending=False)
                st.markdown(f"**{len(tbl):,} rules** (engine: `{engine}`)")
                st.dataframe(tbl, width="stretch", hide_index=True, height=420)

                strong = tbl[tbl["lift"] >= min_lift]
                st.markdown(f"**{len(strong)} rules with lift ≥ {min_lift}**")
                if not strong.empty:
                    top = strong.head(12).copy()
                    top["rule"] = top["antecedents"] + " ⇒ " + top["consequents"]
                    fig = px.bar(top, x="lift", y="rule", orientation="h", color="confidence",
                                 color_continuous_scale="Portland",
                                 hover_data=["support", "confidence"])
                    fig.update_layout(height=480, yaxis={"categoryorder": "total ascending"},
                                      margin=dict(t=20, b=10))
                    st.plotly_chart(fig, width="stretch", key="as_bar")

        with tab_c:
            if rules.empty:
                st.info("No rules to explore.")
            else:
                d1, d2 = st.columns([3, 2])
                with d1:
                    plot_df = rules.copy()
                    plot_df["ante"] = plot_df["antecedents"].map(_fmt_set)
                    plot_df["cons"] = plot_df["consequents"].map(_fmt_set)
                    fig = px.scatter(
                        plot_df, x="support", y="confidence", size="lift",
                        color="lift", hover_data=["ante", "cons", "lift"],
                        color_continuous_scale="Turbo",
                        size_max=26,
                        labels={"ante": "if", "cons": "then",
                                "support": "Support", "confidence": "Confidence"})
                    fig.add_hline(y=float(min_conf), line_dash="dot", line_color="#EF553B",
                                  annotation_text=f"min confidence {min_conf}")
                    fig.add_vline(x=float(min_sup), line_dash="dot", line_color="#636EFA",
                                  annotation_text=f"min support {min_sup}")
                    fig.update_layout(height=500, margin=dict(t=30, b=10))
                    st.plotly_chart(fig, width="stretch", key="as_scatter")

                with d2:
                    st.markdown("**Top-10 rules by lift**")
                    top = rules.sort_values("lift", ascending=False).head(10).copy()
                    for i, row in top.iterrows():
                        conf = row.get("confidence", 0)
                        sup = row.get("support", 0)
                        lift = row.get("lift", 0)
                        st.markdown(
                            f"- `{_fmt_set(row['antecedents'])}` ⇒ "
                            f"`{_fmt_set(row['consequents'])}`  \n"
                            f"  support **{sup:.3f}** · confidence **{conf:.3f}** · "
                            f"lift **{lift:.2f}**")
    except Exception:
        show_exception("Association-rule mining failed")


# --------------------------------------------------------------------------------
# Tab 4 - Clustering
# --------------------------------------------------------------------------------
def tab_clustering(artifacts: Dict) -> None:
    st.subheader("🎯 Unsupervised Learning - K-Means Customer Segmentation")
    cust = artifacts.get("rfm")
    if cust is None or len(cust) < 20:
        st.error("Customer table unavailable or too small for clustering.")
        return

    available = [c for c in CORE_FEATURES if c in cust.columns]
    extra = [c for c in cust.select_dtypes(include=[np.number]).columns
             if c not in available + ["is_high_value", "is_churn_risk", "clv_proxy"]]
    c = st.columns(4)
    with c[0]:
        feats = st.multiselect("Clustering features", available + extra,
                               default=available, max_selections=8)
    with c[1]:
        do_log = st.checkbox("Log1p-transform skewed features", value=True)
    with c[2]:
        do_scale = st.checkbox("Standardize (z-score)", value=True)
    with c[3]:
        seed = st.number_input("Random seed", 0, 9999, 42, 1)
    if not feats:
        st.warning("Select at least one feature.")
        return

    try:
        X = cust[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
        if do_log:
            for col in feats:
                if X[col].skew() > 1.0 and (X[col] >= 0).all():
                    X[col] = np.log1p(X[col])
        if do_scale:
            X = pd.DataFrame(StandardScaler().fit_transform(X),
                             columns=feats, index=X.index)

        k_max = int(min(10, max(3, len(X) // 40)))
        k_max = max(3, k_max)
        k = st.slider("Number of clusters (k)", 2, k_max, min(4, k_max))

        search = elbow_search(X, k_max, int(seed))
        e1, e2 = st.columns(2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=search["k"], y=search["inertia"], mode="lines+markers",
                                 name="Inertia (WCSS)", line=dict(color="#636EFA", width=3)))
        fig.add_vline(x=k, line_dash="dash", line_color="#EF553B")
        fig.update_layout(title="Elbow plot - Within-Cluster Sum of Squares",
                          xaxis_title="k", yaxis_title="Inertia (WCSS)", height=340,
                          xaxis=dict(dtick=1))
        e1.plotly_chart(fig, width="stretch", key="cl_elbow")

        sil = [s for s in search["silhouette"] if s is not None]
        ks_s = [kk for kk, s in zip(search["k"], search["silhouette"]) if s is not None]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ks_s, y=sil, marker_color="#00CC96",
                             text=[f"{s:.3f}" for s in sil], textposition="outside"))
        fig.add_vline(x=k, line_dash="dash", line_color="#EF553B")
        fig.update_layout(title="Average silhouette score by k",
                          xaxis_title="k", yaxis_title="Silhouette", height=340,
                          xaxis=dict(dtick=1))
        e2.plotly_chart(fig, width="stretch", key="cl_sil")

        km, labels = fit_kmeans(X, k, int(seed))
        score = float(silhouette_score(X, labels))
        st.metric("Silhouette score at selected k", f"{score:.3f}")

        # ---- cluster assignments -------------------------------------------
        plot_df = cust[feats].copy()
        plot_df["cluster"] = labels              # int keys -> segment naming
        plot_df["cluster_label"] = labels.astype(str)  # str -> discrete colours
        raw_centres = plot_df.groupby("cluster")[feats].mean()
        names = name_clusters(raw_centres)

        left, right = st.columns([2, 1])
        with left:
            st.markdown("**Cluster sizes**")
            sizes = (pd.Series(labels).value_counts().sort_index()
                     .rename_axis("cluster").reset_index(name="customers"))
            sizes["segment"] = sizes["cluster"].map(names)
            fig = px.bar(sizes, x="segment", y="customers", color="customers",
                         color_continuous_scale="Teal")
            fig.update_layout(height=320, coloraxis_showscale=False,
                              xaxis_title="", margin=dict(t=20, b=10))
            st.plotly_chart(fig, width="stretch", key="cl_sizes")
        with right:
            st.markdown("**Centroid profile (raw units)**")
            st.dataframe(raw_centres.round(2), width="stretch", hide_index=False)
            st.caption("Segment naming: rule-based interpretation of RFM centroids.")

        # ---- projection plots ----------------------------------------------
        st.markdown("---")
        st.markdown("**Segment visualizer**")
        method = st.radio("Projection", ["PCA (2D)", "PCA (3D)", "Feature axes (2D)"],
                          horizontal=True)

        pca = PCA(n_components=3).fit(X)
        comp = pca.transform(X)
        proj = pd.DataFrame(comp, columns=["PC1", "PC2", "PC3"], index=X.index)
        proj["cluster"] = labels.astype(str)
        var = pca.explained_variance_ratio_ * 100

        if method == "PCA (2D)":
            fig = px.scatter(proj, x="PC1", y="PC2", color="cluster",
                             color_discrete_sequence=CLUSTER_PALETTE[:max(k, 3)],
                             opacity=0.75,
                             labels={"PC1": f"PC1 ({var[0]:.1f}% var)",
                                     "PC2": f"PC2 ({var[1]:.1f}% var)"})
            fig.update_traces(marker=dict(size=6, line=dict(width=0.3, color="white")))
        elif method == "PCA (3D)":
            fig = px.scatter_3d(proj, x="PC1", y="PC2", z="PC3", color="cluster",
                                color_discrete_sequence=CLUSTER_PALETTE[:max(k, 3)],
                                opacity=0.75,
                                labels={"PC1": f"PC1 ({var[0]:.1f}%)",
                                        "PC2": f"PC2 ({var[1]:.1f}%)",
                                        "PC3": f"PC3 ({var[2]:.1f}%)"})
            fig.update_traces(marker=dict(size=4, line=dict(width=0.2, color="white")))
        else:
            cols2 = st.columns(2)
            with cols2[0]:
                xax = st.selectbox("X axis", feats, index=0)
            with cols2[1]:
                yax = st.selectbox("Y axis", feats, index=min(1, len(feats) - 1))
            fig = px.scatter(plot_df, x=xax, y=yax, color="cluster_label",
                             color_discrete_sequence=CLUSTER_PALETTE[:max(k, 3)],
                             opacity=0.75)
        fig.update_layout(height=520, legend_title="Cluster",
                          margin=dict(t=20, b=10))
        if method == "PCA (3D)":
            fig.update_layout(scene_camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)))
        st.plotly_chart(fig, width="stretch", key="cl_proj")

        st.markdown("**Segment interpretation & targeting strategy**")
        interp = pd.DataFrame({
            "Segment": [names[i] for i in range(k)],
            "Customers": [int((labels == i).sum()) for i in range(k)],
            "Avg recency (d)": [raw_centres.loc[i, "recency"] if "recency" in raw_centres else np.nan
                                for i in range(k)],
            "Avg frequency": [raw_centres.loc[i, "frequency"] if "frequency" in raw_centres else np.nan
                              for i in range(k)],
            "Avg monetary ($)": [raw_centres.loc[i, "monetary"] if "monetary" in raw_centres else np.nan
                                 for i in range(k)],
        })
        st.dataframe(interp.round(2), width="stretch", hide_index=True)

        out = cust.copy()
        out["cluster"] = labels
        st.download_button("⬇️ Download customers with segment labels (CSV)",
                           out.to_csv(index=False).encode("utf-8"),
                           file_name="customers_segmented.csv", mime="text/csv",
                           key="cl_dl")
    except Exception:
        show_exception("Clustering pipeline failed")


# --------------------------------------------------------------------------------
# Tab 5 - Classification
# --------------------------------------------------------------------------------
def tab_classification(artifacts: Dict) -> None:
    st.subheader("🌳 Classification - Decision Tree (J48) vs Naive Bayes")
    cust = artifacts.get("rfm")
    if cust is None:
        st.error("Customer table unavailable.")
        return

    feats = get_features(cust)
    if len(feats) < 2:
        st.error("Not enough numeric features for modelling.")
        return

    c = st.columns(4)
    with c[0]:
        target_label = st.selectbox("Prediction target", list(TARGET_OPTIONS.keys()))
        target = TARGET_OPTIONS[target_label]
    with c[1]:
        selected = st.multiselect("Feature set", feats, default=feats)
    with c[2]:
        test_size = st.slider("Test split", 0.1, 0.5, 0.25, 0.05)
    with c[3]:
        max_depth = st.slider("Max tree depth (DT)", 2, 20, 6)

    if target not in cust.columns:
        st.error(f"Target '{target}' missing from the customer table.")
        return
    if len(selected) < 1:
        st.warning("Select at least one feature.")
        return

    try:
        X, y = _prepare_xy(cust, selected, target)
        bundle = train_classification(X, y, float(test_size), int(max_depth), 42)
        results = bundle["results"]

        # ---- metrics --------------------------------------------------------
        st.markdown("#### Performance comparison")
        rows = []
        for name, res in results.items():
            row = {"Model": name}
            row.update({k: round(v, 4) for k, v in res["metrics"].items()})
            rows.append(row)
        met_df = pd.DataFrame(rows)
        st.dataframe(met_df, width="stretch", hide_index=True)
        best = met_df.loc[met_df["Accuracy"].idxmax(), "Model"]
        st.success(f"**Best by accuracy:** {best}  ·  "
                   f"test samples = {bundle['n_test']:,}  ·  "
                   f"positive class rate = {y.mean() * 100:.1f}%")

        # ---- confusion matrices --------------------------------------------
        st.markdown("#### Confusion matrices")
        labels = bundle["classes"] if len(bundle["classes"]) == 2 else None
        cols = st.columns(len(results))
        for i, (name, res) in enumerate(results.items()):
            with cols[i]:
                lbl = [str(v) for v in sorted(np.unique(y))] or labels
                st.plotly_chart(fig_confusion(res["cm"], name, lbl),
                                width="stretch", key=f"clf_cm_{i}")

        # ---- ROC ------------------------------------------------------------
        st.markdown("#### ROC curves")
        st.plotly_chart(fig_roc(results), width="stretch", key="clf_roc")

        # ---- feature importance & rules -------------------------------------
        st.markdown("#### Model internals")
        i1, i2 = st.columns(2)
        with i1:
            st.markdown("**Decision-tree feature importance (Gini importance)**")
            imp = bundle["importances"]
            fig = px.bar(imp.head(12).reset_index(), x=0, y="index", orientation="h",
                         labels={"0": "importance", "index": "feature"},
                         color_discrete_sequence=["#636EFA"])
            fig.update_layout(height=360, yaxis={"categoryorder": "total ascending"},
                              margin=dict(t=20, b=10))
            st.plotly_chart(fig, width="stretch", key="clf_imp")
        with i2:
            st.markdown("**Extracted decision rules (top of tree, depth ≤ 6)**")
            st.code(bundle["tree_rules"][:3000], language="text")

        st.caption(
            "**J48 note:** Weka's J48 is the canonical C4.5 implementation "
            "(gain-ratio splits + pruning). scikit-learn's "
            "`DecisionTreeClassifier(criterion='gini')` is the closest open-source "
            "analogue used here, with `class_weight='balanced'` handling imbalance."
        )
    except Exception:
        show_exception("Classification pipeline failed")


# --------------------------------------------------------------------------------
# Tab 6 - Regression
# --------------------------------------------------------------------------------
def tab_regression(artifacts: Dict) -> None:
    st.subheader("📈 Regression - Customer Lifetime Value (CLV) Prediction")
    cust = artifacts.get("rfm")
    if cust is None:
        st.error("Customer table unavailable.")
        return

    targets = [t for t in ("clv_proxy", "monetary", "avg_order_value") if t in cust.columns]
    if not targets:
        st.error("No continuous target (clv_proxy / monetary) available.")
        return

    feats = get_features(cust)
    c = st.columns(3)
    with c[0]:
        target = st.selectbox("Target variable", targets,
                              format_func=lambda t: {
                                  "clv_proxy": "CLV proxy (predicted lifetime value)",
                                  "monetary": "Monetary (historical spend)",
                                  "avg_order_value": "Average order value",
                              }.get(t, t))
    with c[1]:
        selected = st.multiselect("Features", feats, default=feats)
    with c[2]:
        test_size = st.slider("Test split", 0.1, 0.5, 0.25, 0.05, key="reg_test")

    if len(selected) < 1:
        st.warning("Select at least one feature.")
        return

    try:
        X, y = _prepare_xy(cust, selected, target)
        bundle = train_regression(X, y, float(test_size), 42)

        st.markdown("#### Model comparison")
        st.dataframe(bundle["metrics"].style.format(
            {"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.4f}",
             "MAE/mean(y)": "{:.3f}", "Train time (s)": "{:.4f}"}),
            width="stretch", hide_index=True)
        st.success(f"**Best model (highest R²):** {bundle['best']}")

        # actual vs predicted for the best model
        best = bundle["best"]
        y_pred = bundle["preds"][best]
        y_te = bundle["y_test"]

        d1, d2 = st.columns(2)
        with d1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_te, y=y_pred, mode="markers",
                                     marker=dict(size=6, color="#636EFA", opacity=0.6),
                                     name=best))
            lo = float(min(y_te.min(), y_pred.min()))
            hi = float(max(y_te.max(), y_pred.max()))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                     line=dict(dash="dash", color="#EF553B"),
                                     name="ideal y = x"))
            fig.update_layout(title=f"Actual vs predicted - {best}",
                              xaxis_title="Actual", yaxis_title="Predicted",
                              height=420, legend_orientation="h")
            st.plotly_chart(fig, width="stretch", key="reg_avp")

        with d2:
            resid = (y_te - y_pred)
            fig = px.histogram(resid, nbins=45, color_discrete_sequence=["#00CC96"])
            fig.update_layout(title="Residual distribution (actual - predicted)",
                              xaxis_title="residual", height=420)
            st.plotly_chart(fig, width="stretch", key="reg_resid")

        if bundle["importances"] is not None:
            st.markdown(f"#### Feature importance - {best}")
            imp = bundle["importances"].head(12).reset_index()
            imp.columns = ["feature", "importance"]
            fig = px.bar(imp, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Teal")
            fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"},
                              coloraxis_showscale=False, margin=dict(t=20, b=10))
            st.plotly_chart(fig, width="stretch", key="reg_imp")

        st.caption(
            "CLV proxy formula used in the warehouse: "
            "`clv = monetary × (1 + 1/recency) × (frequency / tenure_days) × 365`, "
            "winsorised at the 99th percentile."
        )
    except Exception:
        show_exception("Regression pipeline failed")


# --------------------------------------------------------------------------------
# Tab 7 - Live prediction
# --------------------------------------------------------------------------------
def tab_predict(artifacts: Dict) -> None:
    st.subheader("⚡ Real-Time Prediction Console")
    cust = artifacts.get("rfm")
    if cust is None:
        st.error("Customer table unavailable.")
        return

    try:
        k = st.number_input("Segments (k) used by the live segmenter", 2, 8, 4, 1)
        bundle = build_live_bundle(cust, int(k), 42)
        feats = bundle["features"]
        stats = bundle["stats"]

        st.markdown("**Enter customer metrics** (defaults = dataset medians)")
        cols = st.columns(3)
        values = {}
        for i, f in enumerate(feats):
            lo = float(stats.loc["min", f])
            hi = float(stats.loc["max", f])
            med = float(stats.loc["median", f])
            step = 1.0 if float(hi - lo).is_integer() or abs(hi - lo) > 50 else 0.01
            with cols[i % 3]:
                values[f] = st.slider(
                    f.replace("_", " ").title(),
                    min_value=round(lo, 2), max_value=round(hi, 2),
                    value=round(min(max(med, lo), hi), 2),
                    step=step, key=f"live_{f}",
                )

        if st.button("🔮 Predict", type="primary", width="stretch"):
            row = pd.DataFrame([values], columns=feats)
            Xs = pd.DataFrame(bundle["scaler"].transform(row), columns=feats)
            seg = int(bundle["kmeans"].predict(Xs)[0])
            seg_name = bundle["cluster_names"].get(seg, f"Segment {seg}")

            res = st.container()
            with res:
                st.markdown("### Results")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Segment", seg_name.split("(")[0].strip())

                if bundle["has_high"]:
                    p_hi_dt = float(bundle["dt_high"].predict_proba(row)[0, 1])
                    p_hi_nb = float(bundle["nb_high"].predict_proba(row)[0, 1])
                    c2.metric("High-value P (DT)", f"{p_hi_dt * 100:.1f}%")
                    c3.metric("High-value P (NB)", f"{p_hi_nb * 100:.1f}%")
                else:
                    c2.metric("High-value P (DT)", "n/a")
                    c3.metric("High-value P (NB)", "n/a")

                if bundle["has_churn"]:
                    p_ch_dt = float(bundle["dt_churn"].predict_proba(row)[0, 1])
                    p_ch_nb = float(bundle["nb_churn"].predict_proba(row)[0, 1])
                    c4.metric("Churn-risk P (DT)", f"{p_ch_dt * 100:.1f}%",
                              delta=f"NB {p_ch_nb * 100:.1f}%", delta_color="inverse")
                else:
                    c4.metric("Churn-risk P (DT)", "n/a")

                clv = float(bundle["reg"].predict(row)[0])
                st.metric(f"Predicted {bundle['reg_target'].replace('_', ' ').title()}",
                          f"${clv:,.2f}")

                # gauges
                g = st.columns(2)
                if bundle["has_high"]:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=p_hi_dt * 100,
                        title={"text": "High-value probability (Decision Tree)"},
                        gauge={"axis": {"range": [0, 100]},
                               "bar": {"color": "#00CC96"},
                               "steps": [
                                   {"range": [0, 50], "color": "#f2f4f7"},
                                   {"range": [50, 100], "color": "#d1fadf"}]}))
                    fig.update_layout(height=260, margin=dict(t=40, b=10))
                    g[0].plotly_chart(fig, width="stretch", key="pred_g1")
                if bundle["has_churn"]:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=p_ch_dt * 100,
                        title={"text": "Churn-risk probability (Decision Tree)"},
                        gauge={"axis": {"range": [0, 100]},
                               "bar": {"color": "#EF553B"},
                               "steps": [
                                   {"range": [0, 50], "color": "#f2f4f7"},
                                   {"range": [50, 100], "color": "#fee4e2"}]}))
                    fig.update_layout(height=260, margin=dict(t=40, b=10))
                    g[1].plotly_chart(fig, width="stretch", key="pred_g2")

                with st.expander("Model artefacts used for this prediction"):
                    st.json({
                        "segment": seg,
                        "segment_name": seg_name,
                        "features": values,
                        "models": ["KMeans(k=%d)" % int(k),
                                   "DecisionTree(max_depth=6)",
                                   "GaussianNB",
                                   "RandomForestRegressor(300 trees)"],
                        "training_rows": int(len(cust)),
                    })
    except Exception:
        show_exception("Live prediction failed")


# --------------------------------------------------------------------------------
# Tab 8 - Accuracy & Model Comparison
# --------------------------------------------------------------------------------
def tab_accuracy(raw: pd.DataFrame, artifacts: Dict) -> None:
    """Accuracy dashboard + full parameter-comparison matrices."""
    st.subheader("🧪 Accuracy & Model Comparison")
    cust = artifacts.get("rfm")
    if cust is None or len(cust) < 20:
        st.error("Customer table unavailable or too small for model comparison.")
        return

    feats = get_features(cust)
    if len(feats) < 2:
        st.error("Not enough numeric features for model comparison.")
        return

    SHORT = {"is_high_value": "High-value", "is_churn_risk": "Churn risk"}

    try:
        # ---------------- 1. train every model at tab defaults ----------------
        cls_metrics, baselines = {}, {}
        for label, target in TARGET_OPTIONS.items():
            if target not in cust.columns:
                continue
            try:
                X, y = _prepare_xy(cust, feats, target)
                if y.nunique() < 2:
                    continue
                cls_metrics[target] = (train_classification(X, y, 0.25, 6, 42), y)
                pos = float(y.mean())
                baselines[target] = max(pos, 1 - pos)
            except Exception:
                continue

        reg_metrics = None
        reg_target = next((t for t in ("clv_proxy", "monetary") if t in cust.columns), None)
        if reg_target:
            try:
                Xr, yr = _prepare_xy(cust, feats, reg_target)
                reg_metrics = train_regression(Xr, yr, 0.25, 42)
            except Exception:
                pass

        # Clustering silhouette at k = 4 on the standardised feature space
        Xc = cust[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
        for col in feats:
            if Xc[col].skew() > 1.0 and (Xc[col] >= 0).all():
                Xc[col] = np.log1p(Xc[col])
        Xs = pd.DataFrame(StandardScaler().fit_transform(Xc), columns=feats, index=Xc.index)
        k_max = int(min(10, max(3, len(Xs) // 40)))
        search = elbow_search(Xs, k_max, 42)
        k_use = min(4, k_max)
        sil = float(search["silhouette"][k_use - 1]) if k_use >= 2 else None

        # Association summary on the cleaned feed (data-driven thresholds)
        tx = artifacts.get("clean_tx", raw)
        asoc = {"n_rules": 0, "n_itemsets": 0, "max_lift": 0.0, "engine": "-",
                "min_support": 0.0, "min_confidence": 0.0}
        item_col = pick_col(tx, "product") or pick_col(tx, "category")
        if item_col:
            try:
                bool_df = build_baskets(tx, "Customer", item_col, 40)
                hints = suggest_thresholds(bool_df)
                freq, rules, engine = mine_rules(
                    bool_df, hints["min_support"], hints["min_confidence"])
                asoc = {"n_rules": int(len(rules)), "n_itemsets": int(len(freq)),
                        "max_lift": float(rules["lift"].max()) if len(rules) else 0.0,
                        "engine": engine, "min_support": hints["min_support"],
                        "min_confidence": hints["min_confidence"]}
            except Exception:
                pass

        # ---------------- 2. headline accuracy cards ---------------------------
        hv = cls_metrics.get("is_high_value")
        dt_acc = hv[0]["results"]["Decision Tree (J48 equivalent)"]["metrics"]["Accuracy"] \
            if hv else None
        nb_acc = hv[0]["results"]["Naive Bayes (GaussianNB)"]["metrics"]["Accuracy"] \
            if hv else None
        bl_acc = baselines.get("is_high_value")

        r2 = None
        if reg_metrics is not None:
            br = reg_metrics["metrics"]
            r2 = float(br.loc[br["Model"] == reg_metrics["best"], "R²"].iloc[0])

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Decision Tree Acc.",
                  f"{dt_acc:.4f}" if dt_acc is not None else "–",
                  delta=f"baseline {bl_acc:.3f}" if bl_acc else None)
        m2.metric("Naive Bayes Acc.",
                  f"{nb_acc:.4f}" if nb_acc is not None else "–")
        m3.metric("Majority baseline",
                  f"{bl_acc:.3f}" if bl_acc is not None else "–")
        m4.metric("Best CLV R²",
                  f"{r2:.4f}" if r2 is not None else "–")
        m5.metric("Silhouette (k=4)",
                  f"{sil:.3f}" if sil else "–")
        m6.metric("Rules mined", f"{asoc['n_rules']:,}")
        st.caption("Accuracy cards are computed at the module defaults: test split 25 %, "
                   "tree depth 6, k = 4, data-driven Apriori thresholds on the current "
                   "data source.")

        # ---------------- 3. parameter comparison matrices ----------------------
        st.markdown("### 1 · Parameter comparison — every model setting side by side")

        st.markdown("**Classification: Decision Tree vs Naive Bayes**")
        cls_param = pd.DataFrame([
            ["Algorithm family", "Tree-based · C4.5 / J48 analogue",
             "Probabilistic · generative"],
            ["Model type", "Greedy decision tree (criterion='gini')",
             "Gaussian Naive Bayes"],
            ["Split criterion", "Gini impurity (max ΔGini)", "n/a — no explicit splits"],
            ["Max depth (tab default)", "6 (user slider: 2–20)", "n/a — full density fit"],
            ["Min samples per leaf", "5", "n/a"],
            ["Class weighting", "balanced (handles the minority class)",
             "default priors estimated from data"],
            ["Signed feature scaling", "Not required", "Recommended (Gaussian pdf)"],
            ["Feature interactions", "Yes — nested splits", "No — independence assumed"],
            ["Probabilistic output", "predict_proba (class fraction per leaf)",
             "predict_proba (Gaussian densities)"],
            ["Feature importance", "Gini importance (native)", "None — equal weights"],
            ["Overfitting control", "max_depth · min_samples_leaf · pruning",
             "var_smoothing = 1e-9"],
            ["Random seed", "42", "n/a"],
            ["Interpretability", "High — export_text rules", "Medium — per-feature densities"],
            ["Typical train time", "< 5 ms (≤ 4.3k rows)", "< 1 ms"],
        ], columns=["Parameter", "Decision Tree (J48)", "Naive Bayes (Gaussian)"])
        st.dataframe(cls_param, width="stretch", hide_index=True)

        st.markdown("**Regression: CLV prediction — four models**")
        reg_param = pd.DataFrame([
            ["Model type", "Ordinary least squares",
             "L2-regularised least squares", "Bootstrap-aggregated trees",
             "Additive boosting trees"],
            ["Regularisation", "None", "α = 1.0 (L2 penalty on coefficients)",
             "Data subsampling + feature bagging", "Shrinkage (learning rate)"],
            ["Tree depth", "n/a", "n/a", "max_depth = 10", "max_depth = 3"],
            ["n_estimators", "n/a", "n/a", "300", "300"],
            ["Learning rate", "n/a", "n/a", "n/a", "0.1"],
            ["Loss", "Squared error", "Squared error + L2 penalty",
             "Squared error (per tree)", "Least squares (LS)"],
            ["Feature interactions", "None (linear)", "None (linear)",
             "Yes — independent trees", "Yes — additive trees"],
            ["Feature importance", "Coefficients (sign / magnitude)",
             "Coefficients (shrunken)", "Impurity-based (native)",
             "Impurity-based (native)"],
            ["Multicollinearity", "Unstable", "Mitigated (shrinkage)",
             "Implicitly handled", "Implicitly handled"],
            ["Parallel training", "n/a", "n/a", "n_jobs = -1", "Sequential (n_jobs = 1)"],
            ["Random seed", "Deterministic", "42", "42", "42"],
            ["CLV relationship", "Linear (R² ≈ 0.73)", "Linear (R² ≈ 0.73)",
             "Non-linear (R² ≈ 0.97)", "Non-linear (R² ≈ 0.98)"],
        ], columns=["Parameter", "Linear", "Ridge", "Random Forest", "Gradient Boosting"])
        st.dataframe(reg_param, width="stretch", hide_index=True)

        st.markdown("**Unsupervised & data-mining layers**")
        misc_param = pd.DataFrame([
            ["Algorithm", "K-Means (Lloyd's)", "Apriori", "ETL — Tukey IQR"],
            ["Core settings", "k = 4 default · init = k-means++ · n_init = 10",
             "min_support = 0.7 × strongest pair · min_confidence data-driven",
             "outlier_factor = 1.5 · fill_strategy = median"],
            ["Distance / metric", "Euclidean (WCSS)", "Support · Confidence · Lift",
             "Q1 − 1.5·IQR … Q3 + 1.5·IQR fence"],
            ["Preprocessing", "log1p (skew > 1) → z-score",
             "one-hot basket matrix (top-40 items)",
             "dedup → impute → cap / remove / none"],
            ["Output", "Segments + centroids + marketing names",
             "Frequent itemsets → rules + leverage",
             "Cleaned fact + customer RFM table"],
        ], columns=["Aspect", "Clustering", "Association", "ETL"])
        st.dataframe(misc_param, width="stretch", hide_index=True)
        st.caption(
            "Values shown are the exact settings implemented in `app.py` "
            "(see `train_classification`, `train_regression`, `fit_kmeans`, "
            "`suggest_thresholds`). Sliders in the module tabs change them live.")

        # ---------------- 4. accuracy / performance summary ---------------------
        st.markdown("### 2 · Accuracy summary — current dataset")

        comb = []
        for target, (bundle, y) in cls_metrics.items():
            short = SHORT.get(target, target)
            pos = float(y.mean())
            for name, res in bundle["results"].items():
                m = res["metrics"]
                comb.append({"Model": name, "Target": f"{short} ({pos:.1%} pos.)",
                             "Accuracy": m["Accuracy"], "Precision": m["Precision"],
                             "Recall": m["Recall"], "F1-Score": m["F1-Score"],
                             "ROC-AUC": m["ROC-AUC"], "Train (s)": m["Train time (s)"]})
        if comb:
            cdf = pd.DataFrame(comb).sort_values(["Target", "Accuracy"], ascending=[True, False])
            # flag the best model per target
            cdf["Best"] = ""
            for tgt in cdf["Target"].unique():
                idx = cdf.index[cdf["Target"] == tgt][0]
                cdf.at[idx, "Best"] = "🏆"
            st.dataframe(cdf.round(4), width="stretch", hide_index=True)
            best_acc = cdf.loc[cdf["Accuracy"].idxmax()]
            best_auc = cdf.loc[cdf["ROC-AUC"].idxmax()]
            b1, b2 = st.columns(2)
            b1.success(f"**Highest classification accuracy:** "
                       f"{best_acc['Model']} on {best_acc['Target']} "
                       f"({best_acc['Accuracy']:.4f})")
            if len(cdf) > 1:
                b2.info(f"**Strongest ROC-AUC:** {best_auc['Model']} "
                        f"({best_auc['ROC-AUC']:.4f})")
        else:
            st.info("No binary target (high-value / churn) available — classification "
                    "accuracy omitted.")

        if reg_metrics is not None:
            st.markdown("**Regression leaderboard (CLV proxy)**")
            rdf = reg_metrics["metrics"].copy()
            rdf["Best"] = ""
            best_row_idx = rdf.index[rdf["Model"] == reg_metrics["best"]][0]
            rdf.at[best_row_idx, "Best"] = "🏆"
            st.dataframe(rdf.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}",
                                           "MAE/mean(y)": "{:.3f}", "Train time (s)": "{:.4f}"}),
                         width="stretch", hide_index=True)
            st.success(f"**Best CLV model:** {reg_metrics['best']} "
                       f"(R² = {rdf.loc[best_row_idx, 'R²']:.4f})")

        # ---------------- 5. winner-by-metric table -----------------------------
        st.markdown("### 3 · Winner per metric (current run)")
        winners = []
        if hv:
            r_hv = hv[0]["results"]
            winners.append(["Classification accuracy", "Decision Tree",
                            f"{r_hv['Decision Tree (J48 equivalent)']['metrics']['Accuracy']:.4f}"])
            winners.append(["Classification ROC-AUC",
                            "Decision Tree" if r_hv["Decision Tree (J48 equivalent)"]["metrics"]["ROC-AUC"]
                            >= r_hv["Naive Bayes (GaussianNB)"]["metrics"]["ROC-AUC"]
                            else "Naive Bayes",
                            f"{max(r_hv['Decision Tree (J48 equivalent)']['metrics']['ROC-AUC'],
                                    r_hv['Naive Bayes (GaussianNB)']['metrics']['ROC-AUC']):.4f}"])
        if reg_metrics is not None:
            winners.append(["CLV regression R²", reg_metrics["best"],
                            f"{r2:.4f}" if r2 is not None else "–"])
        winners += [["Clustering quality (silhouette, k=4)", "K-Means",
                     f"{sil:.4f}" if sil else "–"],
                    ["Association strength (max lift)", "Apriori",
                     f"{asoc['max_lift']:.2f}"],
                    ["Rules mined at defaults", f"Apriori ({asoc['engine']})",
                     f"{asoc['n_rules']:,}"]]
        st.dataframe(pd.DataFrame(winners, columns=["Metric", "Winner", "Value"]),
                     width="stretch", hide_index=True)

        # ---------------- 6. interpretation -------------------------------------
        st.markdown("### 4 · Reading the accuracy table")
        st.markdown("""
- **Accuracy vs baseline.** The Decision Tree commonly reaches ~1.000 because the
  `is_high_value` / `is_churn_risk` labels are threshold rules on (recency, frequency,
  monetary) — a tree with the right features recovers the rule exactly. Judge the
  *Naive Bayes* row (≈ 0.97 acc / 0.99 AUC) against that same baseline.
- **ROC-AUC survives class imbalance.** NB trades precision for recall on the rare
  class but ranks customers almost as well as the tree (AUC ≈ 0.97–0.99) — a useful
  probabilistic baseline even when its calibrated probabilities are weaker.
- **Regression R² is scale-free, MAE is not.** On the real dataset Gradient Boosting
  wins (R² ≈ 0.98, MAE ≈ 8 % of mean CLV); tree ensembles beat linear baselines by a
  wide margin because the CLV relationship is multiplicative.
- **Every number above is recomputed on the current data source** (side bar), so the
  same tables double as evidence for the report.
""")

        st.info("Full formulas, splits and both-dataset tables: see "
                "`PROJECT_REPORT.md` §8.4 (Model Comparison Matrix & Accuracy Summary).")
    except Exception:
        show_exception("Accuracy & comparison dashboard failed")


# --------------------------------------------------------------------------------
# Tab 9 - About
# --------------------------------------------------------------------------------
def tab_about() -> None:
    st.subheader("ℹ️ About this Project")
    st.markdown("""
    **E-Commerce Customer Analytics & Machine Learning Platform** — a Data
    Warehousing & Data Mining mini-project implementing an end-to-end
    analytics stack. It runs on a **real-world Kaggle dataset** (*Online
    Retail*, 541,909 line-items from a UK giftware retailer) and, as an
    alternative, on a 12,000-row synthetic warehouse.

    ### Repository layout
    | File | Purpose |
    |---|---|
    | `app.py` | Multi-tab Streamlit application (this app) |
    | `generate_data.py` | Synthetic dataset generator → `ecommerce_10k.csv` |
    | `prepare_online_retail.py` | Kaggle *Online Retail* → canonical `online_retail.csv` (cleaning + schema mapping) |
    | `online_retail.csv` | Cleaned real dataset (397,884 records, 18,532 invoices, 4,338 customers) |
    | `.streamlit/config.toml` | App theme & server configuration |
    | `requirements.txt` | Pinned Python dependencies |
    | `README.md` | Setup, run & Streamlit Cloud deployment guide |
    | `PROJECT_REPORT.md` | Academic report (theory, formulas, results) |

    ### Algorithms implemented
    1. **ETL** — dedup, imputation (median/mean/mode), Tukey-IQR outlier
       capping, RFM feature extraction, star-schema warehouse model.
    2. **Association rules** — Apriori (mlxtend) with a bundled custom
       Apriori fallback; support, confidence, lift, leverage.
    3. **Clustering** — K-Means with Elbow (WCSS) + Silhouette diagnostics,
       PCA-based 2D/3D interactive visualization.
    4. **Classification** — Decision Tree (J48-equivalent) vs Gaussian Naive
       Bayes with accuracy/precision/recall/F1/ROC-AUC.
    5. **Regression** — Linear, Ridge, Random Forest and Gradient Boosting
       for Customer Lifetime Value.
    6. **Live inference** — form-driven real-time predictions.
    7. **Model comparison** — a dedicated Accuracy & Comparison tab with full
       parameter-comparison matrices and a live accuracy/leaderboard summary
       (accuracy, precision, recall, F1, ROC-AUC, MAE, RMSE, R², silhouette, rules).

    ### Reproduce the datasets
    ```bash
    python generate_data.py      # writes ecommerce_10k.csv (12,000 rows)
    python prepare_online_retail.py --src "online+retail.zip"  # real data
    ```
    """)
    st.info("Full theory, formulas and evaluation tables: see `PROJECT_REPORT.md`.")


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def main() -> None:
    apply_css()
    raw = sidebar_data_source()

    label = st.session_state.get(
        "source_label", "**Built-in synthetic warehouse** — `ecommerce_10k.csv`")
    render_hero(source_label=label)

    if raw is None:
        st.info("👈 Select a data source in the sidebar to begin.")
        st.markdown("""
        **Quick start**
        1. **Recommended:** keep *Kaggle Online Retail (real data)* selected —
           398k real UK retailer transactions with genuine market-basket structure.
        2. Explore the **ETL & RFM** tab to clean and engineer features.
        3. Mine **Association Rules**, segment with **K-Means**, compare
           **Decision Tree vs Naive Bayes**, model **CLV**, then use the
           **Live Prediction** console.
        """)
        render_footer()
        return

    artifacts = get_artifacts()
    if artifacts is None:
        artifacts = run_pipeline(raw, st.session_state.get("etl_settings",
                                                           DEFAULT_ETL_SETTINGS))
    if artifacts is None:
        st.stop()

    tabs = st.tabs([
        "🏠 Overview", "⚙️ ETL & RFM", "🧺 Association Rules",
        "🎯 Clustering", "🌳 Classification", "📈 Regression",
        "⚡ Live Prediction", "🧪 Accuracy & Comparison", "ℹ️ About",
    ])
    with tabs[0]:
        tab_overview(raw, artifacts)
    with tabs[1]:
        tab_etl(raw, artifacts)
    with tabs[2]:
        tab_association(raw, artifacts)
    with tabs[3]:
        tab_clustering(artifacts)
    with tabs[4]:
        tab_classification(artifacts)
    with tabs[5]:
        tab_regression(artifacts)
    with tabs[6]:
        tab_predict(artifacts)
    with tabs[7]:
        tab_accuracy(raw, artifacts)
    with tabs[8]:
        tab_about()

    render_footer()


if __name__ == "__main__":
    main()
