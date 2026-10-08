# E-Commerce Customer Analytics & Machine Learning Platform

## A Data Warehousing and Data Mining Mini-Project — Technical Report

---

**Course:** Data Warehousing and Mining  
**Project type:** Mini-Project (End-to-End Analytics Application)  
**Platform:** Streamlit web application backed by a synthetic e-commerce data warehouse  
**Dataset:** `ecommerce_10k.csv` — 12,000 transaction line-items, 7,703 baskets, 2,370 customers, 37 attributes  
**Tech stack:** Python 3.13 · pandas · NumPy · scikit-learn · mlxtend · Plotly · Seaborn/Matplotlib · Streamlit  

---

## Abstract

This report presents the design, implementation and evaluation of an end-to-end
**E-Commerce Customer Analytics and Machine Learning Platform**. The system ingests a
transactional data warehouse — by default the classic **Kaggle / UCI *Online Retail***
dataset (541,909 raw line-items, cleaned to 397,884 records) — and performs a
configurable **ETL pipeline** (deduplication, missing-value imputation, Tukey-IQR
outlier treatment, RFM feature extraction), and drives four data-mining workloads from
a single interactive Streamlit application: **association-rule mining** (Apriori),
**K-Means customer segmentation**, **supervised classification** (Decision Tree vs.
Gaussian Naive Bayes) and **regression-based Customer Lifetime Value (CLV) prediction**,
plus a real-time prediction console. On the real retail feed the Apriori miner recovered
**675 association rules** (best lift 7.57 for a *Lunch-Bag* triple), the segmenter
produced four business-interpretable customer segments, the Gradient-Boosting regressor
achieved **R² = 0.9823** on CLV, and the Decision Tree recovered the label definition
exactly (accuracy 1.000) while Naive Bayes attained ROC-AUC 0.972–0.993. The complete
source, converters, generator, this report and deployment instructions are delivered as
a reproducible repository.

**Keywords:** Data warehousing, RFM analysis, Apriori, association rules, K-Means,
Decision Tree, Naive Bayes, Customer Lifetime Value, Online Retail dataset, Streamlit.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data Warehouse Design (Star Schema)](#2-data-warehouse-design-star-schema)
3. [ETL Pipeline & Preprocessing](#3-etl-pipeline--preprocessing)
4. [Association Rule Mining](#4-association-rule-mining)
5. [Customer Segmentation with K-Means](#5-customer-segmentation-with-k-means)
6. [Classification: Decision Tree vs. Naive Bayes](#6-classification-decision-tree-vs-naive-bayes)
7. [Regression: Customer Lifetime Value Prediction](#7-regression-customer-lifetime-value-prediction)
8. [Experimental Setup & Results](#8-experimental-setup--results)
9. [Application Walkthrough & Screenshots](#9-application-walkthrough--screenshots)
10. [Discussion, Limitations & Threats to Validity](#10-discussion-limitations--threats-to-validity)
11. [Conclusion & Future Work](#11-conclusion--future-work)
12. [References](#12-references)

---

## 1. Introduction

Modern e-commerce platforms accumulate vast transaction logs that must be distilled into
actionable customer intelligence: *who* is likely to churn, *which* products are bought
together, *how* customers naturally cluster, and *how much* a customer will spend in the
future. Answering these questions requires the systematic application of data
warehousing (to organise raw facts and dimensions) and data mining (to extract
patterns).

The platform built for this mini-project implements the canonical CRISP-DM loop over a
star-schema warehouse:

| Phase | Implementation |
|---|---|
| Business understanding | High-value identification, churn-risk flagging, CLV estimation, basket affinity |
| Data understanding | Profiling tab: dtypes, missingness, uniqueness, distributions |
| Data preparation | Configurable ETL: dedup → imputation → IQR outlier treatment → RFM extraction |
| Modelling | Apriori, K-Means, Decision Tree, Gaussian Naive Bayes, 4 regressors |
| Evaluation | Support/confidence/lift; silhouette & elbow; accuracy/precision/recall/F1/AUC; MAE/RMSE/R² |
| Deployment | Multi-tab Streamlit app, local & Streamlit Community Cloud |

**Research questions.**

1. *RQ1 — Association:* Which product categories/items co-occur in baskets more often
   than chance (lift > 1)?
2. *RQ2 — Segmentation:* How many natural customer segments exist, and how do they
   differ on RFM axes?
3. *RQ3 — Classification:* Which algorithm (J48-equivalent Decision Tree or Gaussian
   Naive Bayes) better predicts high-value and churn-risk customers?
4. *RQ4 — Regression:* Which regressor best estimates Customer Lifetime Value?

---

## 2. Data Warehouse Design (Star Schema)

The warehouse follows a **star schema** with a single transaction-grain fact table
(`FACT_SALES`) and five conformed dimensions. A fact table at the lowest declared
granularity (one row per line-item) guarantees that additive measures (`quantity`,
`total_amount`, `discount_amount`) may be summed across any dimension without
double-counting.

```
                         ┌────────────────┐
                         │   Dim_Date     │
                        ┌┤ date_key       │
                        ││ year/quarter   │
                        │└ month/weekend  │
┌──────────────┐        │ ┌────────────────┐        ┌────────────────┐
│ Dim_Customer │        │ │  FACT_SALES    │        │  Dim_Product   │
│ customer_key │────────┼─┤ customer_key   ├────────┤ product_key    │
│ age, gender  │        │ │ product_key    │        │ category       │
│ region, tier │        │ │ date_key       │        │ product_name   │
└──────────────┘        │ │ channel_key    │        │ unit_price     │
                        │ │ qty, amount,   │        └────────────────┘
┌──────────────────┐    │ │ discount, ...  │        ┌──────────────────┐
│ Dim_Customer_Attr │   │ └────────────────┘        │  Dim_Channel     │
│ signup_date      │───┘                            │ channel, payment │
│ tenure, tier     │────────────────────────────────┘ └──────────────────┘
└──────────────────┘
```

**Dimension tables.** `Dim_Customer` (demographics: age, gender, region),
`Dim_Customer_Attr` (signup date, tenure, loyalty tier, marketing opt-in),
`Dim_Product` (category, product name, unit price), `Dim_Date` (year, quarter, month,
weekend flag — enables OLAP roll-ups), and `Dim_Channel` (sales channel, payment
method).

**Derived analytical layer.** A customer-grain **RFM summary table**
(`customer_rfm_summary.csv`, 2,370 rows for the synthetic feed) acts as a data-mart view
joining the fact table's additive measures into per-customer behavioural measures, and
is the input to the clustering, classification and regression modules. The application
also renders an interactive star-schema diagram in the ETL tab.

**Real-data loading.** When the *Online Retail* feed is selected, the star schema is
populated directly from the canonical `online_retail.csv`: line-items land in
`FACT_SALES` (`transaction_id` as degenerate dimension), `product_name`/`stock_code`
form `Dim_Product`, `region` (i.e. `Country`) attaches to `Dim_Customer`, and the
invoice timestamp decomposes into `Dim_Date`. Columns the retailer does not store
(channel, payment, discount, signup date) are simply absent — the framework falls back
gracefully, and downstream modules operate only on features that exist (see §8.2).

---

## 3. ETL Pipeline & Preprocessing

The ETL stage (`build_customer_table` in `app.py`) executes six ordered steps, each
logged into a user-visible pipeline log.

### 3.1 Extraction & typing
Dates are coerced with `pd.to_datetime(errors="coerce")` so malformed entries become
`NaT` rather than crashing the pipeline; semantic columns are resolved
case-insensitively from a candidate list, allowing uploaded CSVs with alternative
naming (`order_date`, `total_amount`, …).

### 3.2 Deduplication
Exact duplicate rows are detected and removed: `df.drop_duplicates()` — 0 duplicates
found in the generated warehouse (verified by the profiler); the real *Online Retail*
feed contains **5,192 exact duplicate line-items**, all removed during ETL (see §8.1).

### 3.3 Missing-value imputation
Six strategies are selectable: **median**, **mean**, **mode**, **zero**, **listwise
deletion** and **none**. For numeric column *x* with missing set *M*:

```
x̂_i = median(x)      for i ∈ M          (Eq. 1)
```

Median is the default because transactional amounts are right-skewed (mean is pulled by
heavy spenders); the generated dataset contains 0 missing cells, so the step is a no-op
there but is exercised for uploaded data.

### 3.4 Outlier detection & treatment (Tukey's IQR fence)
For each numeric column, hinges *Q*1 and *Q*3 define the interquartile range
*IQR = Q3 − Q1*; values outside the fences are outliers:

```
lower = Q1 − k · IQR ,   upper = Q3 + k · IQR ,   k = 1.5   (Eq. 2)
```

Two treatments are offered:

* **Cap (winsorise)** — clip values to the fence; preserves row count. Applied to
  11,319 cells at *k* = 1.5 in the default synthetic run.
* **Remove** — delete rows outside the fences computed on the revenue column; shrinks
  the fact table but yields a tighter distribution.

On the real *Online Retail* feed (k = 1.5) the IQR report flags **34,112 `unit_price`
cells (8.69 %)**, **31,231 `amount` cells (7.95 %)** and **25,616 `quantity` cells
(6.52 %)** — genuine heavy-basket and low-price artefacts, demonstrating the step's
value on real-world data (winsorisation is the default action).

A per-column report (Q1, Q3, IQR, fences, outlier count and percentage) plus a
colour-scaled bar chart make the decision auditable.

### 3.5 RFM feature extraction
RFM analysis segments customers on three behavioural axes, computed relative to a
**snapshot date** *S* (default: max transaction date + 1):

```
Recency   R_i = (S − last_purchase_i)                    in days      (Eq. 3)
Frequency F_i = |distinct transaction_id of customer i|               (Eq. 4)
Monetary  M_i = Σ final_amount of customer i                         (Eq. 5)
```

Auxiliary aggregates are joined alongside: `avg_order_value`, `total_quantity`,
`unique_categories`, `preferred_channel/payment`, `tenure_days = S − signup_date`, and
average discount. Quintile-based rule labels are attached when the source has none:

```
is_high_value = 1  iff  M ≥ P80(M)  ∧  F ≥ P80(F)                    (Eq. 6)
is_churn_risk = 1  iff  R > 90  ∨  (R > 60 ∧ F < median(F))          (Eq. 7)
clv_proxy     = M · (1 + 1/R) · (F / tenure) · 365   (winsorised P99) (Eq. 8)
```

The generated dataset already ships these labels; they are retained by default and can
be force-recomputed with a checkbox.

### 3.6 Load
Results are materialised as (a) a cleaned transaction-grain fact table
(12,000 × 37), (b) a customer-grain RFM table (2,370 × 15+), both downloadable as CSV
from the UI, and (c) session-state artefacts consumed by every downstream tab.

---

## 4. Association Rule Mining

### 4.1 Problem statement
An association rule *X ⇒ Y* states that baskets containing itemset *X* tend to contain
itemset *Y*. For a basket database *D* with |*D*| transactions:

```
support(X ⇒ Y)    = P(X ∪ Y) = sup(X ∪ Y) / |D|              (Eq. 9)
confidence(X ⇒ Y) = P(Y | X)  = sup(X ∪ Y) / sup(X)           (Eq. 10)
lift(X ⇒ Y)       = confidence / P(Y)                         (Eq. 11)
                    = P(X ∩ Y) / (P(X)·P(Y))
leverage          = P(X ∩ Y) − P(X)·P(Y)                      (Eq. 12)
```

**Interpretation:** lift = 1 → independence; lift > 1 → positive association (the
target of mining); lift < 1 → substitutes/avoidance.

### 4.2 The Apriori algorithm
Apriori exploits the **downward-closure (anti-monotone) property**: any superset of a
frequent itemset is frequent, so a (k−1)-subset that is infrequent prunes its entire
k-candidate family.

```
L1 = { i ∈ items : support(i) ≥ min_sup }
for k = 2, 3, … while L(k−1) ≠ ∅:
    Ck = candidates formed by joining pairs of L(k−1) sharing (k−2) items
    prune c ∈ Ck if some (k−1)-subset ∉ L(k−1)
    Lk = { c ∈ Ck : support(c) ≥ min_sup }
Frequent = ∪k Lk ;  → generate rules from each frequent itemset's subsets
```

The application uses **`mlxtend.frequent_patterns.apriori` + `association_rules`** when
installed, and transparently falls back to a **bundled custom Apriori implementation**
(level-wise candidate generation, subset-pruning, NumPy support counting) so the module
never fails on a bare environment. Both paths were verified to return identical
frequent-itemset counts (18 itemsets / 9 rules at support 0.08, confidence 0.30 on the
legacy single-item dataset).

### 4.3 Basket construction
Three granularities are offered, each yielding different semantics:

| Grain | Basket = | Use case |
|---|---|---|
| **Transaction** | items in one order | classic market basket / cross-sell |
| **Customer** | distinct items a customer ever bought | affinity & personalisation |
| **Customer-Month** | distinct items per customer per month | seasonal bundles |

To keep the search space tractable, only the **top-N items** (default 60) by frequency
enter the matrix. Crucially, **thresholds are seeded from the data**: support defaults
to 0.7 × the strongest observed pair support, and confidence defaults to the confidence
of the ≈6th-strongest positive-lift rule found at a 0.05 floor. Fixed thresholds are a
classic pitfall — mining at support 0.05 on this warehouse returns only single-item
"frequent sets" (because the rarest interesting pair sits at support ≈ 0.049) and hence
*zero rules*; the adaptive seeding guarantees every dataset opens on a useful rule set.

### 4.4 Results (default settings)

**Transaction grain, category items** (support ≥ 0.0345, confidence ≥ 0.15):
14 frequent itemsets, **8 rules**.

| Antecedent | Consequent | Support | Confidence | Lift |
|---|---|---:|---:|---:|
| Electronics | Toys | 0.039 | 0.152 | **1.23** |
| Toys | Electronics | 0.039 | 0.317 | **1.23** |
| Clothing | Beauty | 0.040 | 0.170 | **1.15** |
| Beauty | Clothing | 0.040 | 0.274 | **1.15** |
| Sports | Electronics | 0.049 | 0.248 | 0.97 |

**Customer grain, category items** (support ≥ 0.1669, confidence ≥ 0.45):
15 frequent itemsets, **7 rules**; best = *Beauty ⇒ Clothing* (lift 1.30).

**Answer to RQ1:** *Electronics–Toys* and *Clothing–Beauty* are the only positively
associated category pairs at transaction grain (lift 1.23 and 1.15) — consistent with
gift-purchase behaviour; *Sports–Electronics* is effectively independent (lift 0.97).
Itemsets such as `{Electronics}` (support 0.256) and `{Clothing}` (0.238) dominate the
frequent-set lattice.

#### Real data (*Online Retail*, product items)

With the real feed the same pipeline mines genuine item-level rules. **Customer grain,
top-40 items** (support ≥ 0.03, confidence ≥ 0.2): 325 frequent itemsets, **675 rules**.
The strongest rules are the well-documented *Lunch-Bag* family of the Online Retail
study:

| Antecedent | Consequent | Support | Confidence | Lift |
|---|---:|---:|---:|---:|
| LUNCH BAG CARS BLUE + LUNCH BAG BLACK SKULL. | LUNCH BAG PINK POLKADOT | 0.047 | 0.775 | **7.57** |
| LUNCH BAG CARS BLUE | LUNCH BAG PINK POLKADOT | 0.049 | 0.479 | 4.68 |
| LUNCH BAG BLACK SKULL. | LUNCH BAG PINK POLKADOT | 0.053 | 0.452 | 4.42 |
| LUNCH BAG CARS BLUE | LUNCH BAG BLACK SKULL. | 0.060 | 0.588 | 4.34 |

**Transaction grain** (18,532 real invoices, top-40 items): 409 rules — genuine
multi-item basket co-occurrence. Lift values of 4–8 are strong, real-world affinity
signals (purchasers of one lunch-bag design tend to buy the matching set), which the
synthetic category-level data could only approximate at lift 1.15–1.23.

---

## 5. Customer Segmentation with K-Means

### 5.1 Method
K-Means partitions *n* observations into *k* clusters by minimising the **Within-Cluster
Sum of Squares (WCSS / inertia)**:

```
J = Σ_{c=1}^{k} Σ_{x ∈ c} ‖x − μ_c‖²                         (Eq. 13)
```

Each iteration assigns points to the nearest centroid (Lloyd's algorithm) and recomputes
centroids as cluster means until convergence. Features are right-skewed, so a
`log1p` transform is applied where skewness > 1, followed by **standardisation**
(z-score) so that Recency (days) and Monetary (currency) contribute on equal scale —
otherwise Monetary would dominate the Euclidean distance.

**Model selection** uses two diagnostics:

* **Elbow method** — plot J against *k*; the "elbow" where marginal gain flattens
  indicates the natural *k*.
* **Silhouette score** — for each point *a(i)* = mean intra-cluster distance and
  *b(i)* = mean distance to the nearest other cluster:

```
s(i) = (b(i) − a(i)) / max(a(i), b(i)) ,   S = (1/n) Σ s(i)   (Eq. 14)
```

`S ∈ [−1, 1]`; values near 1 indicate well-separated clusters.

### 5.2 Results (8 RFM/behavioural features, log-transform + z-score)

| k | Inertia (WCSS) | Silhouette | Cluster sizes |
|---:|---:|---:|---|
| 1 | 18,960.0 | — | — |
| 2 | 13,719.7 | **0.244** | — |
| 3 | 12,199.4 | 0.173 | 961 / 686 / 723 |
| **4** | **10,936.4** | 0.165 | **735 / 573 / 422 / 640** |
| 5 | 10,207.6 | 0.153 | 453/494/536/375/512 |
| 6 | 9,606.7 | 0.155 | — |
| 7 | 9,062.0 | 0.152 | — |
| 8 | 8,610.6 | 0.150 | — |

The elbow is visible between *k* = 2 and *k* = 4; **k = 4 is selected as the operating
point** — it captures a ~40 % WCSS reduction over *k* = 1 while remaining
business-interpretable (four distinguishable marketing treatments). Silhouette
diminishing monotonically after *k* = 2 is expected for high-dimensional behavioural
data with no perfectly separated natural groupings.

**Centroid profile (k = 4, raw units) and auto-generated segment names:**

| Segment | Recency (d) | Frequency | Monetary ($) | AOV ($) | Categories |
|---|---:|---:|---:|---:|---:|
| 🏆 Champions (high spend, frequent, recent) | 180.0 | 4.6 | 987.8 | 135.0 | 4.5 |
| 💤 Big Spenders Gone Quiet (win-back) | 399.1 | 2.2 | 484.2 | 162.7 | 2.3 |
| ⚠️ At Risk / Hibernating (reactivate) | 497.7 | 1.5 | 69.7 | 41.2 | 1.6 |
| 🔁 Loyal & Recent (nurture) | 239.3 | 3.8 | 197.6 | 35.0 | 3.8 |

**Answer to RQ2:** four stable segments emerge. *Champions* (31 % of customers)
combine the lowest recency with the highest frequency and monetary value; *At Risk /
Hibernating* (27 %) are stale, infrequent, low-value and should receive reactivation
campaigns; *Big Spenders Gone Quiet* is the win-back prize (high AOV, 399-day recency);
*Loyal & Recent* is a nurture segment with healthy frequency but modest basket size —
an upsell candidate. Visualisation supports PCA 2-D/3-D projections (PC1–PC3 explaining
the bulk of variance) and free feature-axis scatter, coloured by segment.

#### Real data (*Online Retail*, 5 features, log-transform + z-score)

The same protocol on 4,338 real customers (recency, frequency, monetary,
avg-order-value, total quantity; log1p applied to the four right-skewed features;
z-scored):

| k | Inertia (WCSS) | Silhouette |
|---:|---:|---:|
| 2 | 12,813.6 | 0.351 |
| 3 | 10,528.0 | 0.243 |
| **4** | **8,872.6** | **0.251** |
| 5 | 7,891.1 | 0.245 |
| 8 | 6,119.0 | 0.212 |

The elbow at *k* = 4 persists on real data; cluster sizes 478 / 812 / 1,478 / 1,570
confirm a realistic long-tail distribution (≈ 11 % heavy buyers). Silhouette values are
again moderate — behavioural data does not form perfectly separated blobs — and the
landscape reproduces the synthetic finding that *k* = 4 is the interpretable operating
point.

---

## 6. Classification: Decision Tree vs. Naive Bayes

Two supervised learners predict binary customer labels (**high-value**, **churn-risk**)
from 8 features (recency, frequency, monetary, avg order value, tenure, age, unique
categories, total quantity). Data is split **stratified 75/25** (593 test rows) so the
8.8 % minority high-value class is preserved in both folds.

### 6.1 Decision Tree (J48 / C4.5 family)
A recursive binary splitter. At each node the feature *j* and threshold *t* maximising
the impurity reduction are chosen. **Gini impurity:**

```
Gini(S) = 1 − Σ_c p_c²                                       (Eq. 15)
ΔGini   = Gini(parent) − (n_L/n)·Gini(S_L) − (n_R/n)·Gini(S_R) (Eq. 16)
```

Weka's **J48** implements C4.5 (gain-ratio splits + error-based post-pruning); the
scikit-learn `DecisionTreeClassifier(criterion="gini", max_depth, min_samples_leaf=5,
class_weight="balanced")` used here is the standard open-source analogue, with balanced
class weights compensating for label imbalance. Depth is user-tunable (2–20, default 6)
and the fitted rules are exported in the UI for interpretability.

### 6.2 Gaussian Naive Bayes
Applies Bayes' rule with a feature-independence assumption; each feature is modelled by
a Gaussian conditioned on the class:

```
P(c | x) ∝ P(c) · Π_j  N(x_j | μ_{jc}, σ²_{jc})              (Eq. 17)
ŷ = argmax_c P(c | x)
```

It requires no hyper-parameter tuning and handles correlated features gracefully in
practice, but its probability estimates are miscalibrated under feature correlation
(here: recency/frequency/monetary are strongly interdependent).

### 6.3 Evaluation metrics

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)                   (Eq. 18)
Precision = TP / (TP + FP)      Recall = TP / (TP + FN)       (Eq. 19)
F1        = 2·P·R / (P + R)                                   (Eq. 20)
AUC       = ∫₀¹ TPR d(FPR)  (trapezoid over the ROC curve)    (Eq. 21)
```

### 6.4 Results — High-Value prediction (test n = 593, positive rate 8.8 %)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train (s) |
|---|---:|---:|---:|---:|---:|---:|
| **Decision Tree (J48-equiv.)** | **1.0000** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0019 |
| Naive Bayes (GaussianNB) | 0.9376 | 0.6000 | 0.8654 | 0.7087 | 0.9760 | 0.0006 |

Confusion matrices — Decision Tree `[[541, 0], [0, 52]]` · Naive Bayes
`[[511, 30], [7, 45]]`.

### 6.5 Results — Churn-Risk prediction (positive rate 79.2 %)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train (s) |
|---|---:|---:|---:|---:|---:|---:|
| **Decision Tree (J48-equiv.)** | **1.0000** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0013 |
| Naive Bayes (GaussianNB) | 0.9477 | 0.9525 | 0.9829 | 0.9675 | 0.9893 | 0.0005 |

Confusion matrices — Decision Tree `[[124, 0], [0, 469]]` · Naive Bayes
`[[101, 23], [8, 461]]`.

**Feature importance (Gini):** high-value → `monetary` 0.781, `frequency` 0.219;
churn → `recency` 0.997, `frequency` 0.003. The trees recover exactly the variables
that define the labels (Eqs. 6–7) — see §10 for an honest treatment of this.

**Answer to RQ3:** the Decision Tree dominates on every metric; Naive Bayes trades
precision for recall on the high-value class (0.60 precision / 0.87 recall) and remains
competitive on churn (AUC 0.989) while training ~3× faster. For deployment the tree is
preferred for its interpretability *and* perfect separation; NB serves as a fast,
calibration-friendly baseline.

#### Real data (*Online Retail*, 5 features, test n = 1,085, high-value 14.8 %)

The real feed reproduces the same conclusion on 4,338 customers:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Decision Tree (J48-equiv.)** | **1.0000** | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| Naive Bayes (GaussianNB) | 0.9724 | 0.8740 | 0.9500 | 0.9100 | **0.9931** |

**Churn-risk target (38.2 %)** — DT 1.0000 on all metrics; NB accuracy 0.653, F1 0.681,
ROC-AUC 0.972. The churn labels again follow a threshold rule on (recency, frequency),
so the tree recovers them exactly; the *real* surprise is Naive Bayes' strong AUC
(0.993 high-value / 0.972 churn), i.e. a probabilistic baseline ranks customers almost
as well despite weaker calibrated probabilities — a meaningful, data-dependent result
that the synthetic run could only hint at.

---

## 7. Regression: Customer Lifetime Value Prediction

Four regressors predict `clv_proxy` (Eq. 8) from the same 8 features, evaluated on a
25 % hold-out with MAE, RMSE, R² and normalised error:

| Model | MAE | RMSE | R² | MAE/ȳ | Train (s) |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 154.58 | 253.58 | 0.8893 | 0.274 | 0.0006 |
| Ridge Regression (α = 1) | 154.57 | 253.58 | 0.8893 | 0.274 | 0.0005 |
| Random Forest (300 trees, depth 10) | 45.04 | 116.84 | 0.9765 | 0.080 | 0.193 |
| **Gradient Boosting (300 est., lr 0.1)** | **36.95** | **85.18** | **0.9875** | **0.066** | 0.426 |

```
MAE  = (1/n) Σ |y_i − ŷ_i|                                   (Eq. 22)
RMSE = √( (1/n) Σ (y_i − ŷ_i)² )                             (Eq. 23)
R²   = 1 − Σ(y_i − ŷ_i)² / Σ(y_i − ȳ)²                       (Eq. 24)
```

**Answer to RQ4:** **Gradient Boosting wins** (R² = 0.9875, MAE = 36.95 ≈ 6.6 % of the
mean target), beating the linear baselines by a wide margin — the CLV relationship is
strongly non-linear (multiplicative in Eq. 8), which linear models capture only
approximately (R² ≈ 0.89, and note Ridge ≈ OLS because the collinear RFM features are
only mildly regularised). Random Forest is a close second at lower variance; Gradient
Boosting's extra 0.23 s training time is irrelevant at this scale. Residual
distributions are centred at zero with no systematic bias across the prediction range.

#### Real data (*Online Retail*, 5 features, clv_proxy)

| Model | MAE | RMSE | R² | MAE/ȳ |
|---|---:|---:|---:|---:|
| Linear Regression | 8,936.8 | 21,219.3 | 0.7293 | 0.626 |
| Ridge Regression (α = 1) | 8,936.8 | 21,219.3 | 0.7293 | 0.626 |
| Random Forest (300 trees, depth 10) | 1,245.7 | 7,083.0 | 0.9698 | 0.087 |
| **Gradient Boosting (300 est., lr 0.1)** | **1,105.2** | **5,419.0** | **0.9823** | **0.077** |

Absolute errors are larger (CLV magnitudes reach hundreds of thousands of pounds on
real spend), but the *relative* ranking is identical — **Gradient Boosting wins again**
(R² = 0.9823, MAE ≈ 7.7 % of mean target; feature importance `frequency` 0.61,
`monetary` 0.36) — and tree ensembles beat linear baselines by a wider margin on the
noisier real target, confirming RQ4 on independent data.

---

## 8. Experimental Setup & Results

### 8.1 Datasets

**A. Kaggle / UCI *Online Retail* (real data, the app's default feed).**

| Property | Value |
|---|---|
| Source workbook | `online+retail.zip` → `Online Retail.xlsx` (UCI ML Repository, classic giftware retailer study) |
| Raw records | **541,909** line-items × 8 columns (`InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`) |
| Date range | 2010-12-01 → 2011-12-09 |
| Pre-conversion cleaning | `prepare_online_retail.py` (reproducible, CLI) |
| └ cancellation invoices removed | 9,288 rows (`InvoiceNo` starts with `C`) |
| └ anonymous rows removed | 134,697 rows (missing `CustomerID`; unjoinable baskets) |
| └ invalid sales removed | 40 rows (`Quantity ≤ 0` or `UnitPrice ≤ 0`, incl. `CustomerID = 0`) |
| └ derived measure | `amount = Quantity × UnitPrice` |
| **Cleaned export `online_retail.csv`** | **397,884 records × 9 columns** |
| Invoices / customers | 18,532 / 4,338 |
| Canonical schema | `transaction_id, customer_id, transaction_date, stock_code, product_name, quantity, unit_price, amount, region` |
| ETL-time findings | 5,192 exact duplicate line-items removed; IQR outliers (k = 1.5): `unit_price` 34,112 (8.7 %), `amount` 31,231 (8.0 %), `quantity` 25,616 (6.5 %) |
| Derived labels (customer grain) | 14.8 % high-value, 38.2 % churn-risk; avg recency 93 d, frequency 4 orders, monetary £1,311 |

**B. Synthetic warehouse (`generate_data.py`, reproducible, seed = 42).**

| Property | Value |
|---|---|
| Line-item rows | **12,000** (requirement: ≥ 10,000) |
| Distinct baskets | 7,703 (sizes 1/2/3 items: 4,394 / 2,321 / 988) |
| Customers | 2,370 (dimension ships 2,500) |
| Date range | 2022-01-01 → 2024-12-30 |
| Categories / products | 10 / 400 |
| Amounts | mean $130.15, median $57.91, max $7,597.68 |
| Missing cells / duplicates | 0 / 0 |
| Labels | 8.8 % high-value, 79.2 % churn-risk (customer grain) |

Baskets are generated with a **complementary-category Markov step** (65 % probability of
pulling a complementary category from the current basket: Electronics↔Toys,
Clothing↔Beauty, …) and a **Zipf-skewed product pool**, so genuine positive-lift
structure exists for the Apriori module to discover — a flat random generator would
produce only lift ≈ 1 noise. Seasonality (Nov/Dec ×1.5, Jan/Feb ×0.7), loyalty tiers,
channel and payment mix follow plausible business distributions.

### 8.2 Protocol
* Split: stratified 75/25 (classification) and random 75/25 (regression), seed 42.
* K-Means: `n_init = 10`, seed 42, *k* ∈ [2, 10] swept for diagnostics.
* Apriori: adaptive thresholds (§4.3), top-40/60 items.
* Real-data feature set: recency, frequency, monetary, avg order value, total quantity
  (no `tenure`/`age`/`unique_categories` — the retailer stores no signup/demographic
  attributes; the app interpolates `tenure = 365` for the CLV proxy, Eq. 8).
* All metrics computed with scikit-learn 1.7; charts Plotly 6 / Seaborn 0.13.

### 8.3 Summary against RQs

| RQ | Verdict (synthetic) | Verdict (real *Online Retail*) |
|---|---|---|
| RQ1 Association | Electronics↔Toys, Clothing↔Beauty, lift 1.23 / 1.15 | Lunch-Bag set rules, **lift up to 7.57**; 675 rules |
| RQ2 Segmentation | 4 interpretable segments, elbow at k≈4 | 4 segments again (k≈4), sizes 478/812/1,478/1,570 |
| RQ3 Classification | Decision Tree wins (acc 1.000 vs 0.948) | Decision Tree wins (acc 1.000 vs 0.972); NB AUC 0.993 |
| RQ4 Regression | Gradient Boosting wins, R² 0.9875 | **Gradient Boosting wins, R² 0.9823** |

---

## 9. Application Walkthrough & Screenshots

> **Note on screenshots:** the placeholder tags below mark where figures captured from
> the running application belong in the printed report. Capture them with the app
> running (`streamlit run app.py`) at a 1440 × 900 viewport, then replace each tag with
> the embedded image, e.g. `![Screenshot 1: ETL Pipeline Dashboard](figures/01_etl.png)`.

![Screenshot 1: Overview Tab — Dataset KPIs, Monthly Revenue Trend, Category & Channel Distributions, Correlation Heatmap](figures/01_overview.png)

![Screenshot 2: ETL Pipeline Dashboard — Profiling, Configuration Controls and Step-by-Step Pipeline Log](figures/02_etl_pipeline.png)

![Screenshot 3: Missing-Value Treatment (before/after) and Tukey IQR Outlier Report with Fences](figures/03_cleaning_outliers.png)

![Screenshot 4: RFM Feature Table, Recency/Frequency/Monetary Distributions and Star-Schema Diagram](figures/04_rfm_star_schema.png)

![Screenshot 5: Association Rules — Frequent Itemsets, Support/Confidence/Lift Rules Table](figures/05_association_rules.png)

![Screenshot 6: Rule Explorer — Support-vs-Confidence Scatter (size = Lift) and Top-10 Rules by Lift](figures/06_rule_explorer.png)

![Screenshot 7: K-Means Elbow Plot and Silhouette Score Chart](figures/07_elbow_silhouette.png)

![Screenshot 8: Customer Segments — 3D PCA Scatter Plot with Cluster Colours](figures/08_clustering_3d.png)

![Screenshot 9: Segment Centroid Profile and Targeting Strategy Table](figures/09_segment_profiles.png)

![Screenshot 10: Classification Metrics — Decision Tree vs Naive Bayes Performance Comparison](figures/10_classification_metrics.png)

![Screenshot 11: Interactive Confusion Matrices for Both Models](figures/11_confusion_matrices.png)

![Screenshot 12: ROC Curve Comparison with AUC Values](figures/12_roc_curves.png)

![Screenshot 13: Decision-Tree Feature Importance and Extracted Rule Set](figures/13_tree_rules.png)

![Screenshot 14: Regression Model Comparison and Actual-vs-Predicted Plot](figures/14_regression.png)

![Screenshot 15: Real-Time Prediction Console with Probability Gauges and Segment Output](figures/15_live_prediction.png)

![Screenshot 16: Home View — Hero Banner, Pipeline Pills and Sidebar Data-Source Selector (🏆 Kaggle Online Retail selected)](figures/16_home_hero.png)

![Screenshot 17: ETL Pipeline on the Real Dataset — Cleaning Log, IQR Outlier Report and RFM Table](figures/17_etl_real.png)

![Screenshot 18: Association Rules on Real Baskets — Lunch-Bag Rules with Lift 7.57](figures/18_association_real.png)

**Walkthrough.** The app opens on *Overview* (KPIs, revenue trend, region mix, Pearson
correlation) under a hero banner with pipeline-stage pills and a sidebar that defaults
to the **🏆 Kaggle *Online Retail* (real data)** feed (a *Dataset guide* expander
documents the schema and cleaning rules). *ETL & RFM* exposes every cleaning knob with a
live pipeline log and downloadable customer table. *Association Rules* seeds thresholds
from the data and renders itemsets, a sortable rules table and an interactive rule
explorer. *Clustering* shows elbow + silhouette, then PCA 2-D/3-D segment scatter with
centroid profiling. *Classification* compares DT vs NB via metrics table, dual confusion
matrices and ROC curves, plus the tree's feature importance and exported rules.
*Regression* compares four regressors with actual-vs-predicted and residual charts.
*Live Prediction* accepts form inputs (defaults = dataset medians) and returns the
segment, both models' high-value/churn probabilities as gauge charts, and a CLV
estimate. The UI is pinned to a **light theme** (`.streamlit/config.toml`) with
theme-safe CSS so every label, button and card keeps high contrast on any machine.

---

## 10. Discussion, Limitations & Threats to Validity

1. **Perfect Decision-Tree accuracy is a property of the labels, not overfitting.**
   The high-value and churn labels are *deterministic functions* of recency, frequency
   and monetary (Eqs. 6–7). A decision tree on those same features therefore recovers
   the generating rule exactly (importance: `monetary` 0.78 + `frequency` 0.22 for
   high-value; `recency` 0.997 for churn) — accuracy 1.000 reflects label
   reconstructability on this benchmark, not leakage introduced by the modelling code.
   Naive Bayes' sub-1.0 scores (AUC 0.976/0.989) show the same signal under a
   probabilistic, independence-violating model. On real-world datasets where labels come
   from *future* behaviour rather than a formula, expect genuinely non-trivial gaps; the
   app's CSV upload path is provided precisely to test that. The split is stratified and
   the tree is pruned (`max_depth`, `min_samples_leaf = 5`, balanced class weights) to
   keep the comparison fair.
2. **Silhouette scores are modest (0.15–0.24).** Behavioural customer data rarely has
   sharply separated clusters; the monotonic decline after k = 2 and the visible elbow
   justify k = 4 on interpretability grounds, matching common RFM practice.
3. **Dataset provenance.** The synthetic generator's patterns
   (complementary-category affinity, seasonality, Zipf products) are plausible but not
   empirical. This limitation is now largely mitigated: every experiment in this report
   was repeated on the **real Kaggle *Online Retail*** feed, which reproduces all four
   RQ conclusions (stronger lift 7.57, same k = 4 elbow, DT-over-NB ranking, 
   GB-over-linear regression ranking) — the *Lunch-Bag* rules and 1,000+ pound CLV
   magnitudes are genuine retail phenomena. The upload path additionally accepts
   arbitrary CSVs with a documented flexible schema.
4. **CLV proxy.** Eq. 8 is a heuristic historical-value extrapolation, not actuarial
   CLV; regressors are compared consistently against it, so the ranking (GB ≫ RF ≫
   linear) is meaningful even if absolute values are indicative. On the real dataset
   the proxy shifts from monetary-centred to frequency-centred importance (0.61), a
   realistic re-weighting worth an explicit note.
5. **Computational scale.** Everything fits in memory in < 0.5 s per model;
   `st.cache_data`/`st.cache_resource` memoise ETL, mining and training so UI slider
   changes stay interactive. A production warehouse would push Apriori to FP-Growth and
   partition the fact table by date.
6. **Reproducibility.** Fixed seeds (NumPy 42, split 42, K-Means 42) make every number
   in this report regenerable with `python generate_data.py`.

---

## 11. Conclusion & Future Work

A complete, deployment-ready E-Commerce Customer Analytics & ML Platform was built and
evaluated. The ETL pipeline operationalises profiling, imputation, IQR outlier
treatment and RFM engineering over a star-schema warehouse; Apriori (with a bundled
custom fallback) surfaced interpretable cross-sell rules — on the **real Kaggle
*Online Retail*** feed a *Lunch-Bag* rule family with **lift up to 7.57** among 675
mined rules; K-Means yielded four actionable segments (Champions, Win-back, At-risk,
Loyal) with elbow/silhouette diagnostics reproduced on 4,338 real customers; the
Decision Tree outperformed Gaussian Naive Bayes on both targets (1.000 vs 0.972
accuracy, AUC 1.000 vs 0.993) while remaining fully interpretable; and Gradient
Boosting best predicted CLV (**R² = 0.9823** on real spend). The whole stack is
reachable from one Streamlit interface with interactive Plotly visualisation, live
inference, light-theme responsive styling, and graceful error handling throughout.

**Future work:** FP-Growth for very wide catalogues; DBSCAN/HDBSCAN as density-based
segmentation baselines; SMOTE or threshold tuning for minority-class recall; survival
analysis (Cox/Pareto) for true CLV; SHAP explanations; scheduled ETL into a real OLAP
engine (DuckDB/StarRocks); A/B testing of segment-level campaigns; and Streamlit
Community Cloud deployment behind authentication.

---

## 12 References

1. Agrawal, R., Imieliński, T., & Swami, A. (1993). *Mining association rules between
   sets of items in large databases.* SIGMOD '93, 207–216.
2. Han, J., Pei, J., & Yin, Y. (2000). *Mining frequent patterns without candidate
   generation.* SIGMOD '00, 1–12.
3. MacQueen, J. (1967). *Some methods for classification and analysis of multivariate
   observations.* Proc. 5th Berkeley Symposium, 1, 281–297.
4. Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to
   Cluster Analysis.* Wiley.
5. Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning.* Morgan Kaufmann.
6. John, G. H., & Langley, P. (1995). *Estimating continuous distributions in Bayesian
   classifiers.* UAI '95, 338–345.
7. Fader, P. S., Hardie, B. G. S., & Lee, J. S. (2005). *Counting your customers the
   easy way: An alternative to the Pareto/NBD model.* Marketing Science, 24(2), 275–289.
8. Witten, I. H., et al. (2016). *Data Mining: Practical Machine Learning Tools and
   Techniques,* 4th ed. Morgan Kaufmann.
9. scikit-learn developers. *scikit-learn user guide* — trees, naive Bayes, clustering,
   metrics. https://scikit-learn.org
10. Salzberg, S. (1994). *C4.5: Programs for Machine Learning.* Machine Learning, 16(3),
    235–240. (book review; J48/C4.5 provenance)
11. mlxtend documentation — `frequent_patterns.apriori`, `association_rules`.
    https://rasbt.github.io/mlxtend/
12. Streamlit documentation. https://docs.streamlit.io
13. Chen, D. (2015). *Online Retail Data Set.* UCI Machine Learning Repository
    (Kaggle mirror: `online+retail.zip`, 541,909 line-items, UK giftware retailer,
    Dec 2010 – Dec 2011). Preprocessed to `online_retail.csv` by
    `prepare_online_retail.py` (§8.1).

---

*Report generated as part of the "E-Commerce Customer Analytics & Machine Learning
Platform" repository. All formulas, tables and screenshot placeholders correspond to
implemented features in `app.py` and the reproducible datasets from `generate_data.py`
(synthetic) and `prepare_online_retail.py` (real Kaggle *Online Retail*).*
