# 🛍️ E-Commerce Customer Analytics & Machine Learning Platform

**A Data Warehousing & Data Mining mini-project** — an end-to-end, production-ready
Streamlit application covering ETL, RFM engineering, association-rule mining,
K-Means segmentation, classification, regression and real-time prediction on the
classic **Kaggle / UCI "Online Retail" dataset** (the recommended, default feed)
plus a 12,000-row synthetic warehouse as an alternative.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

| # | Module | What it does |
|---|---|---|
| 1 | **Data Warehouse ETL** | Profiling, dedup, missing-value imputation (median/mean/mode/zero/drop), Tukey-IQR outlier detection with cap/remove, step-by-step pipeline log, star-schema diagram, CSV export |
| 2 | **RFM Engineering** | Recency / Frequency / Monetary extraction at customer grain + high-value, churn-risk and CLV labels |
| 3 | **Association Rules** | Apriori market-basket analysis (mlxtend, with a bundled custom Apriori fallback) → Support, Confidence, Lift, Leverage; 3 basket grains; data-adaptive thresholds; interactive rule explorer |
| 4 | **Clustering** | K-Means with Elbow (WCSS) + Silhouette diagnostics, auto segment naming, interactive 2D/3D Plotly scatter (PCA or feature axes) |
| 5 | **Classification** | Decision Tree (J48-equivalent) vs Gaussian Naive Bayes on high-value / churn targets, with metrics table, interactive confusion matrices and ROC curves |
| 6 | **Regression** | Linear, Ridge, Random Forest and Gradient Boosting models for Customer Lifetime Value (MAE / RMSE / R²) |
| 7 | **Live Prediction** | Form-based real-time console: segment, class probabilities (gauges) and CLV estimate |
| 8 | **Data source flexibility** | 🏆 **Kaggle *Online Retail* (real data, default)**, built-in `ecommerce_10k.csv`, **or** upload your own CSV (flexible column naming) |

---

## 📁 Repository structure

```
.
├── app.py                  # Multi-tab Streamlit application (main entry point)
├── generate_data.py        # Synthetic dataset generator → ecommerce_10k.csv
├── prepare_online_retail.py# Kaggle "Online Retail" → online_retail.csv (clean + map)
├── ecommerce_10k.csv       # 12,000-row transaction fact table (generated)
├── online_retail.csv       # Cleaned real dataset: 397,884 records / 4,338 customers
├── customer_rfm_summary.csv# Customer-grain RFM table (generated)
├── customer_demographics.csv # Customer dimension (generated)
├── .streamlit/config.toml  # App theme & server configuration (light theme)
├── requirements.txt        # Python dependencies
├── PROJECT_REPORT.md       # Academic report (theory, formulas, results)
└── README.md               # This file
```

---

## 🚀 Quick start (local)

### 1. Prerequisites
* Python **3.9 – 3.13**
* `pip` (and optionally `venv`)

### 2. Clone & install

```bash
# from the project directory
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. Generate the datasets

**Synthetic feed (alternative to the real dataset):**

```bash
python generate_data.py
```

This writes `ecommerce_10k.csv` (**12,000 rows × 37 columns**), plus
`customer_rfm_summary.csv` and `customer_demographics.csv`. The generator is seeded
(seed = 42), so output is reproducible. *Skip this step if the CSVs already exist —
the app auto-generates them on first run if they are missing.*

**✅ Recommended — real Kaggle data (*Online Retail*):**

The cleaned export `online_retail.csv` already ships in the repo. To regenerate it
from your downloaded workbook/zip (541,909 line-items, Dec 2010 – Dec 2011):

```bash
python prepare_online_retail.py --src "online+retail.zip"
# or: python prepare_online_retail.py --src "/path/to/Online Retail.xlsx"
```

The converter applies standard data-quality rules and maps the raw schema
(`InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID,
Country`) onto the app's warehouse schema — cancellations (`InvoiceNo` starting
with `C`) and anonymous rows are dropped, invalid quantity/price rows removed, and
`amount = Quantity × UnitPrice` derived (full log in the report, §8).

### 4. Run the app

```bash
streamlit run app.py
```

The browser opens automatically at **http://localhost:8501** and the app loads the
🏆 *Kaggle Online Retail (real data)* feed by default. Switch feeds any time in the
sidebar (real data / built-in synthetic / upload).

> **Tip:** force a specific address/port with
> `streamlit run app.py --server.port 8502 --server.headless true`

### 5. (Optional) Verify dependencies

```bash
pip install -r requirements.txt
python -c "import streamlit, sklearn, mlxtend, plotly; print('OK')"
```

---

## 🧪 Using the application

1. **Sidebar** — by default the app runs on **🏆 Kaggle *Online Retail* (real data)**;
   switch to *Built-in dataset* or **upload your own CSV** as needed.
   * Transaction-level CSVs need: a customer id, a date column, an amount column
     (aliases like `order_date`, `total_amount`, `user_id` are auto-detected).
   * Customer-level CSVs need: `recency`, `frequency`, `monetary`
     (labels are derived automatically if absent).
2. **⚙️ ETL & RFM** — configure imputation/outlier/RFM options, click
   **🚀 Run ETL Pipeline**, inspect the pipeline log and charts, download the
   customer-grain table.
3. **🧺 Association Rules** — pick item type (category/product), basket grain
   (Customer / Customer-Month / Transaction) and top-N items; thresholds are seeded
   from the data, then explore itemsets, the rules table and the rule explorer.
4. **🎯 Clustering** — choose features, *k*, and projection (PCA 2D/3D or feature
   axes); review elbow/silhouette and segment profiles.
5. **🌳 Classification** — select target (high-value / churn), features, test split
   and tree depth; compare Decision Tree vs Naive Bayes.
6. **📈 Regression** — model CLV and compare regressors.
7. **⚡ Live Prediction** — enter metrics (defaults = dataset medians) and hit
   **🔮 Predict**.

---

## ☁️ Deploy to Streamlit Community Cloud

1. **Push the repository to GitHub** (see `.gitignore` already provided):

   ```bash
   git init
   git add .
   git commit -m "E-Commerce Customer Analytics & ML Platform"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

2. Open **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.

3. Click **“New app” → “From a repo”**, then select your repository, branch (`main`)
   and the **Main file path**: `app.py`.

4. Click **“Deploy”**. Streamlit installs `requirements.txt` automatically and the
   app goes live at `https://<your-repo>-<random>.streamlit.app`.

### Deploy notes
* **Include the CSVs** in the repo (synthetic ~3 MB, real `online_retail.csv` ~39 MB)
  *or* rely on the built-in auto-generation: if `ecommerce_10k.csv` is absent,
  `app.py` imports `generate_data.py` and creates it on first request. For the real
  dataset, run `prepare_online_retail.py` locally and commit the resulting CSV, or
  run the converter as a build step.
* Optional `packages.txt` for system dependencies (rarely needed here).
* To keep the app private on Community Cloud, use Streamlit's
  [authentication options](https://docs.streamlit.io/streamlit-community-cloud/deploy).
* **Secrets/config** belong in `.streamlit/secrets.toml` (git-ignored by default).

---

## 📦 Dependencies (`requirements.txt`)

```
streamlit>=1.36.0
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0
scikit-learn>=1.3.0
mlxtend>=0.23.0
plotly>=5.15.0
seaborn>=0.12.0
matplotlib>=3.7.0
scipy>=1.11.0
```

`mlxtend` powers Apriori; if it is unavailable the app automatically falls back to a
bundled custom Apriori implementation, so nothing breaks.

---

## 📊 Datasets

### Kaggle / UCI *Online Retail* (real data, default)

| Property | Value |
|---|---|
| Source | `online+retail.zip` → `Online Retail.xlsx` (ug.es / UCI Machine Learning Repository) |
| Raw workbook | 541,909 line-items × 8 columns, Dec 2010 – Dec 2011, UK giftware retailer |
| Cleaning | cancellations ✂ dropped (9,288), missing `CustomerID` dropped (134,697), invalid qty/price dropped (40), `amount = Quantity × UnitPrice` derived |
| Cleaned export | **397,884 records** · 18,532 invoices · 4,338 customers · 9 columns |
| Schema | `transaction_id, customer_id, transaction_date, stock_code, product_name, quantity, unit_price, amount, region` |
| Structure | genuine multi-item baskets → real Apriori rules, RFM, segmentation |

### Built-in synthetic warehouse (`ecommerce_10k.csv`)

| Property | Value |
|---|---|
| Rows | **12,000** line items (≥ 10,000 required) |
| Baskets | 7,703 (1–3 items each) |
| Customers | 2,370 (dimension ships 2,500) |
| Columns | 37 (transaction, product, customer, RFM, labels) |
| Date range | 2022-01-01 → 2024-12-30 |
| Labels | `is_high_value` (8.8 %), `is_churn_risk` (79.2 %), `clv_proxy` |
| Missing / duplicates | 0 / 0 |

The generator injects **complementary-category basket affinity**
(Electronics↔Toys, Clothing↔Beauty, …), Zipf-skewed product popularity and
Nov/Dec seasonality, so association mining and segmentation discover real structure.

---

## 📚 Report & docs

* **`PROJECT_REPORT.md`** — full academic report: star schema, ETL theory,
  Apriori/RFM/K-Means/Decision-Tree/Naive-Bayes formulas, evaluation tables,
  screenshot placeholders, limitations and references.

---

## 🧩 Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: ecommerce_10k.csv` | `python generate_data.py` |
| Sidebar says "Kaggle Online Retail … needs conversion" | Run `python prepare_online_retail.py --src "online+retail.zip"` (or point at the `.xlsx`) |
| `ModuleNotFoundError: mlxtend` | `pip install -r requirements.txt` |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Charts not rendering | `pip install -U plotly streamlit` |
| Upload rejected | Ensure the CSV has customer id + date + amount **or** recency/frequency/monetary columns |
| “No rules found” in Apriori | Lower *minimum support* / confidence sliders (thresholds are seeded from data; very sparse item spaces need support < 0.01) |
| Buttons/text appear washed out | Delete stale `.streamlit/config.toml` overrides from older versions, or re-run with `streamlit run app.py` (theme is pinned to light) |

---

## 📄 License

MIT — free to use for academic and personal projects.
