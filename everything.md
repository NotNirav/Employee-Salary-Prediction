# EVERYTHING: Technical Architecture & Project Reference

> **Concise, exhaustive architectural compendium for the Employee Salary Prediction system (DEVL Project).**  
> Covers mathematical foundations, domain feature engineering, statistical hypothesis testing, model benchmarking, codebase walkthrough, and runtime behavior.

---

## 1. Executive Summary & Objective

- **Domain:** Human Resources Analytics / Applied Predictive Modeling (DEVL Project, SY PCCOE).
- **Core Objective:** Map structured employee demographic and professional attributes ($\mathbf{x}$) to continuous annual compensation ($y \in \mathbb{R}^+$) using a robust 5-stage machine learning and statistical pipeline.
- **Key Breakthrough:** Overcoming the 193 unique raw job title sparsity problem by introducing a 9-domain functional role classification, a 5-tier seniority extraction hierarchy, and interaction terms.
- **Key Outcome:** Random Forest Regressor achieving $R^2 \approx 0.8693$, $\text{RMSE} \approx \$19,401.50$, and $\text{MAE} \approx \$12,408.90$ across an 80/20 train-test partition, yielding a **30.3% error reduction** over baseline linear regression.

---

## 2. Directory Tree & Artifact Mapping

```text
Employee-Salary-Prediction/
├── requirements.txt            # Package dependencies
├── README.md                   # Project overview and user guide
├── everything.md               # [This file] Complete technical architecture reference
├── main.py                     # CLI router & pipeline orchestrator
├── test_smoke.py               # Automated 45-point regression & integrity smoke test
│
├── data/
│   ├── raw/
│   │   └── Salary_Data.csv     # Raw dataset (6,704 rows x 6 features)
│   └── processed/
│       ├── clean_salary_data.csv    # Cleaned human-readable dataset (1,780 rows)
│       └── encoded_salary_data.csv  # Numeric model-ready dataset (17 features)
│
├── stages/
│   ├── __init__.py             # Package initializer
│   ├── helpers.py              # Domain categorization, seniority extraction & path config
│   ├── stage1_load_summary.py  # Stage 1: Data ingestion, schema inspection & null audit
│   ├── stage2_preprocess.py    # Stage 2: Cleaning, feature engineering & one-hot encoding
│   ├── stage3_eda.py           # Stage 3: Exploratory data analysis & 9 analytical plots
│   ├── stage4_stats.py         # Stage 4: Statistical inference (t-tests, ANOVA) & RF importance
│   └── stage5_models.py        # Stage 5: Regressors, evaluation, parity plots & inference engine
│
└── outputs/
    ├── models/
    │   ├── salary_rf_model.joblib   # Serialized scikit-learn best model bundle
    │   └── model_metrics.txt        # Textual ledger of model evaluation & benchmark
    └── plots/                       # 13 high-resolution (120 DPI) analytical plots
        ├── 01_salary_distribution.png
        ├── 02_salary_boxplot.png
        ├── 03_age_experience_distribution.png
        ├── 04_experience_vs_salary.png
        ├── 05_age_vs_salary.png
        ├── 06_education_vs_salary.png
        ├── 07_job_category_vs_salary.png
        ├── 08_seniority_vs_salary.png
        ├── 09_gender_vs_salary.png
        ├── 10_correlation_heatmap.png
        ├── 11_feature_importance.png
        ├── 12_actual_vs_predicted.png
        └── 13_model_r2_comparison.png
```

---

## 3. Technology Stack & Architectural Roles

| Technology | Role | Specific Usage in Project |
| :--- | :--- | :--- |
| **Python 3.14+** | Runtime | Core execution environment; UTF-8 stream handling on Windows. |
| **Pandas** | ETL / DataFrames | Schema validation, record deduplication, conditional filtering, imputation, feature assembly. |
| **NumPy** | Vectorized Math | Array metrics calculation, log/square roots, array slicing, ratio features. |
| **Scikit-Learn** | ML Pipeline | `StandardScaler`, `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, metrics. |
| **SciPy (stats)** | Inferential Stats | Welch’s two-sample $t$-test (`ttest_ind(equal_var=False)`), One-Way ANOVA (`f_oneway`). |
| **Matplotlib** | Static Rendering | Headless plotting via `Agg` backend; plot canvas generation across 13 analytical figures. |
| **Seaborn** | Statistical Viz | Distribution KDEs, ordered boxplots, correlation heatmaps. |
| **Joblib** | Model Persistence | Serialization and deserialization of the trained Random Forest model bundle. |

---

## 4. End-to-End Pipeline Architecture

```text
[Raw CSV: data/raw/Salary_Data.csv] (6,704 rows)
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ 1. Data Ingestion & Auditing (Stage 1)                 │
│   • Schema check, null counting, scrape duplication    │
└────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ 2. Data Cleaning & Feature Engineering (Stage 2)       │
│   • Normalize strings: Bachelor's, Master's, PhD       │
│   • Drop NaN & non-positive salaries                   │
│   • Deduplicate rows (4,912 removed)                   │
│   • Prune rare classes (Gender == 'Other')             │
│   • Feature Engineering: Job Category (9 domains)      │
│   • Feature Engineering: Seniority (levels 0 to 4)     │
│   • Feature Engineering: Experience_per_Age ratio      │
│   • Interaction: Exp_x_Education                       │
│   • Output: clean_salary_data.csv & encoded_salary_... │
└────────────────────────────────────────────────────────┘
            │ Cleaned & Encoded DataFrames (1,780 rows)
            ├──────────────────────────────┐
            ▼                              ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│ 3. Visual & Stat EDA     │   │ 4. Hypothesis Testing    │
│    (Stage 3)             │   │    (Stage 4)             │
│   • 9 Analytical PNGs    │   │   • Welch's t-test (Edu) │
│   • Median salary tables │   │   • One-Way ANOVA (Edu)  │
│                          │   │   • One-Way ANOVA (Jobs) │
│                          │   │   • Random Forest Import.│
│                          │   │   • Correlation Heatmap  │
└──────────────────────────┘   └──────────────────────────┘
            │                              │
            └──────────────┬───────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 5. Model Training & Evaluation (Stage 5)               │
│   • 80/20 Train-Test Partition (Seed = 42)             │
│   • Models: Linear Regression, Decision Tree, RF       │
│   • Evaluation: MAE, MSE, RMSE, R² (Train vs Test)     │
│   • Visuals: Parity plot (12) & R² comparison (13)     │
│   • Persistence: salary_rf_model.joblib & metrics.txt  │
└────────────────────────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│ Model Artifact           │   │ Inference Gateway        │
│   • salary_rf_model.joblib│   │   • python main.py predict│
│   • model_metrics.txt    │   │   • Interactive terminal CLI│
└──────────────────────────┘   └──────────────────────────┘
```

---

## 5. Mathematical Formulations & Statistical Engine

### 5.1 Multiple Linear Regression (OLS)
Given feature vector $\mathbf{x} = [x_1, x_2, \dots, x_p]^T$:
$$\hat{y} = \beta_0 + \sum_{j=1}^{p} \beta_j x_j + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
Estimated via Ordinary Least Squares (OLS) minimizing the residual sum of squares:
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### 5.2 Decision Tree & Random Forest Regressor
A regression tree recursively partitions the feature space into $M$ disjoint regions $R_1, R_2, \dots, R_M$. The prediction for region $R_m$ is the mean of the training observations falling into that region:
$$\hat{c}_m = \frac{1}{N_m} \sum_{i \in R_m} y_i$$
Split points are selected to minimize Mean Squared Error (MSE):
$$\mathcal{L}(j, s) = \sum_{i \in R_1(j, s)} (y_i - \hat{c}_1)^2 + \sum_{i \in R_2(j, s)} (y_i - \hat{c}_2)^2$$

A **Random Forest** aggregates $B$ de-correlated trees (bagging with feature subspace sampling):
$$\hat{y}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$$

### 5.3 Inferential Hypothesis Testing

#### 5.3.1 Welch's Two-Sample $t$-test (Unequal Variances)
Used to evaluate whether Master's degree holders earn statistically significantly higher compensation than Bachelor's degree holders without assuming equal group variances:
$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$
Degrees of freedom calculated using the Welch–Satterthwaite equation:
$$\nu \approx \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2 / n_1)^2}{n_1 - 1} + \frac{(s_2^2 / n_2)^2}{n_2 - 1}}$$
- **Result:** $t = -15.56, p < 0.0001 \implies$ **Reject $H_0$**.  
  Bachelor's Mean: \$91,211 ($n=768$) vs. Master's Mean: \$127,373 ($n=567$).

#### 5.3.2 One-Way Analysis of Variance (ANOVA)
Tests whether the mean salary differs across $K$ groups:
$$F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} = \frac{\frac{1}{K-1} \sum_{k=1}^K n_k (\bar{y}_k - \bar{y})^2}{\frac{1}{N-K} \sum_{k=1}^K \sum_{i=1}^{n_k} (y_{ki} - \bar{y}_k)^2}$$
- **Education Level ANOVA:** $F = 363.42, p < 0.0001 \implies$ **Reject $H_0$**.
- **Job Category ANOVA:** $F = 18.23, p < 0.0001 \implies$ **Reject $H_0$**.

---

## 6. Resolving the 193 Job Titles Sparsity Dilemma

| Feature | Preprocessing Action | Technical & Statistical Rationale |
| :--- | :--- | :--- |
| `Years of Experience` | Continuous numeric | Primary linear driver of compensation ($r = +0.82$). |
| `Age` | Continuous numeric | High correlation with experience ($r \approx 0.98$); proxies career maturity. |
| `Gender` | Binary dummy (`Gender_Male`) | Captures baseline shifts across demographics. |
| `Education Level` | Ordinal encoding (`0` to `3`) | Monotonic tiering confirmed via boxplots and ANOVA ($F = 363.42$). |
| `Job Title` (Raw) | **Transformed** | **193 unique titles.** Direct one-hot encoding would inject 190+ sparse indicator columns into a 1,424-sample training set, creating severe rank deficiency, extreme coefficient variance, and overfitting. |
| `Job Category` | **9-domain keyword mapping** | Aggregates titles into 9 robust functional domains (`Data/Analytics`, `Engineering/Software`, `HR`, `Marketing`, `Sales`, `Finance/Accounting`, `Design/Creative`, `Operations/Admin`, `Management/Executive`). Zero samples assigned to 'Other'. |
| `Seniority` | **5-level hierarchical extraction** | Maps title prefixes/keywords to ordinal tiers (0=Junior, 1=Mid, 2=Senior, 3=Lead/Manager, 4=Director/Exec). |
| `Exp_x_Education` | Interaction term | Captures differential wage returns to experience conditioned on educational attainment. |

---

## 7. Model Evaluation & Benchmark Ledger

| Model Architecture | Train $R^2$ | Test $R^2$ | MAE ($) | RMSE ($) | Overfitting Gap | Best Model? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.8121 | 0.8130 | $17,686.42 | $23,206.22 | ~0.000 | Baseline Linear |
| **Decision Tree** | 0.9049 | 0.8365 | $14,283.97 | $21,698.88 | 0.0684 | Non-linear Tree |
| **Random Forest** | **0.9715** | **0.8693** | **$12,408.90** | **$19,401.50** | 0.1022 | **Best Overall** |

### Benchmark Against Old Baseline Model

| Metric | Old Baseline (Dropped Job Titles) | New Best Model (Random Forest + Feature Engineering) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Test $R^2$** | 0.7311 | **0.8693** | **+0.1382 (+18.9%)** |
| **RMSE ($)** | $27,828.00 | **$19,401.50** | **-$8,426.50 (30.3% error reduction)** |
| **MAE ($)** | $21,246.00 | **$12,408.90** | **-$8,837.10 (41.6% error reduction)** |

---

## 8. Complete Breakdown of 13 Visualizations

| Plot File | Chart Type & Target | Core Analytical Insight |
| :--- | :--- | :--- |
| `01_salary_distribution.png` | Histogram + KDE | Right-skewed distribution; primary density in \$50,000–\$110,000 range with senior tail reaching \$250,000. |
| `02_salary_boxplot.png` | Univariate Boxplot | Identifies absence of destructive outliers post-deduplication; natural spread across executive compensation. |
| `03_age_experience_distribution.png` | Dual Histograms | Workforce demographic heavily concentrated between ages 25–40 with 2–15 years of tenure. |
| `04_experience_vs_salary.png` | Bivariate Scatter | Strong positive linear trend ($r = +0.82$); steep earnings ramp across early-to-mid career. |
| `05_age_vs_salary.png` | Bivariate Scatter | Steady compensation progression with advancing employee chronological age. |
| `06_education_vs_salary.png` | Ordered Boxplot | Monotonic salary expansion across tiers: High School < Bachelor's < Master's < PhD. |
| `07_job_category_vs_salary.png` | Categorical Boxplot | Executive/Management and Engineering/Data roles command the highest median compensation brackets. |
| `08_seniority_vs_salary.png` | Boxplot by Tier | Clear stepwise upward shifts in compensation across seniority levels 0 (Junior) through 4 (Director/VP). |
| `09_gender_vs_salary.png` | Demographic Boxplot | Illustrates baseline gender salary distributions in the sample. |
| `10_correlation_heatmap.png` | Pearson Heatmap | Highlights high collinearity between Age and Experience ($r \approx 0.98$); validates retention for tree models. |
| `11_feature_importance.png` | Horizontal Bar Chart | Ranks Random Forest feature importances; confirms Years of Experience and Age account for majority of variance. |
| `12_actual_vs_predicted.png` | Parity Plot ($y$ vs. $\hat{y}$) | Test set predictions cluster tightly along ideal 45° line; demonstrates strong generalization. |
| `13_model_r2_comparison.png` | Comparative Bar Chart | Visualizes train vs. test $R^2$ across Linear Regression, Decision Tree, and Random Forest. |

---

## 9. Operational CLI Cheatsheet

```powershell
# 1. Environment Installation
pip install -r requirements.txt

# 2. Execute Full 5-Stage Workflow End-to-End
python main.py

# 3. Execute Individual Stages
python main.py stage 1    # Data Ingestion & Summary
python main.py stage 2    # Cleaning, Feature Engineering & Encoding
python main.py stage 3    # Exploratory Data Analysis & Plots 01-09
python main.py stage 4    # Statistical Inference, Tests & Plots 10-11
python main.py stage 5    # Regressors, Evaluation, Plots 12-13 & Inference

# 4. Interactive Terminal Salary Predictor
python main.py predict

# 5. Automated 45-Point Smoke Test Suite
python test_smoke.py
```
