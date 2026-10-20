# Employee Salary Prediction

**Data Exploration and Visualization Laboratory (DEVL) Mini-Project**  
*Department of Computer Engineering, Pimpri Chinchwad College of Engineering (PCCOE)*

---

## 1. Project Overview & Problem Statement

Annual employee compensation depends on a multifaceted combination of professional qualifications, accumulated tenure, domain specialization, and seniority.

**Problem:** Can we accurately predict employee salaries from structured demographic and occupational attributes while extracting actionable statistical insights from workforce patterns?

**Key Innovations & Results:**
- Standardized and sanitized real-world salary data across 6,704 raw employee records.
- Solved the **193-title high-cardinality dilemma** by engineering a 9-domain functional role taxonomy and a 5-tier seniority hierarchy.
- Validated degree wage premiums and categorical salary distributions via **Welch's $t$-test** and **One-Way ANOVA**.
- Built and benchmarked three regression architectures (Multiple Linear Regression, Decision Tree, Random Forest Regressor).
- Achieved **$R^2 = 0.8693$** (Random Forest), reducing prediction error by **30.3% RMSE** and **41.6% MAE** over the traditional baseline.
- Provided a modular CLI to execute any stage individually, the full pipeline, or an interactive terminal predictor.

---

## 2. Directory Structure

```text
Employee-Salary-Prediction/
├── requirements.txt            # Python dependencies (pandas, scikit-learn, etc.)
├── README.md                   # Complete project documentation and lab guide
├── everything.md               # Technical architecture & mathematical compendium
├── main.py                     # Master CLI orchestrator (train / stage / predict)
├── test_smoke.py               # Automated 45-point test verification suite
│
├── data/
│   ├── raw/
│   │   └── Salary_Data.csv     # Raw dataset (6,704 rows x 6 features)
│   └── processed/
│       ├── clean_salary_data.csv    # Cleaned human-readable dataset (1,780 rows)
│       └── encoded_salary_data.csv  # Numeric model-ready dataset with engineered features
│
├── stages/
│   ├── __init__.py             # Package initializer
│   ├── helpers.py              # Domain keyword categorization and path hierarchy
│   ├── stage1_load_summary.py  # Stage 1: Dataset ingestion, schema inspection & null audit
│   ├── stage2_preprocess.py    # Stage 2: Cleaning, feature engineering & one-hot encoding
│   ├── stage3_eda.py           # Stage 3: Exploratory data analysis & 9 analytical plots
│   ├── stage4_stats.py         # Stage 4: Statistical inference (t-tests, ANOVA) & RF importance
│   └── stage5_models.py        # Stage 5: Regressors, evaluation, parity plots & inference engine
│
└── outputs/
    ├── models/
    │   ├── salary_rf_model.joblib   # Persisted best model bundle (Random Forest)
    │   └── model_metrics.txt        # Textual ledger of model metrics and baseline comparison
    └── plots/                       # 13 high-resolution (120 DPI) visualization charts
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

## 3. Five-Stage Pipeline Breakdown

### Stage 1: Data Ingestion & Summary (`stages/stage1_load_summary.py`)
- Reads raw CSV `Salary_Data.csv` (6,704 rows, 6 features).
- Audits column types, missing value distributions, and duplicate records (4,912 scrape duplicates identified).
- Highlights inconsistencies in `Education Level` strings and extreme sparsity across 193 unique `Job Title` entries.

### Stage 2: Data Preprocessing & Feature Engineering (`stages/stage2_preprocess.py`)
- **Target Sanitization**: Drops 5 rows with missing or non-positive salaries.
- **String Harmonization**: Unifies `"Bachelor's Degree"` $\to$ `"Bachelor's"`, `"Master's Degree"` $\to$ `"Master's"`, `"phD"` $\to$ `"PhD"`.
- **Deduplication**: Removes 4,912 duplicate rows.
- **Subgroup Pruning**: Drops 7 rows where `Gender == 'Other'` due to insufficient sample size for statistical validity.
- **Missing Value Imputation**: Median for continuous features (`Age`, `Years of Experience`); mode for categoricals.
- **Domain Feature Engineering**:
  - `Job Category`: Solves the 193-title sparsity problem by mapping titles into 9 functional categories using keyword matching (`Data/Analytics`, `Engineering/Software`, `HR`, `Marketing`, `Sales`, `Finance/Accounting`, `Design/Creative`, `Operations/Admin`, `Management/Executive`).
  - `Seniority`: Extracts hierarchical career tier (0=Junior, 1=Mid, 2=Senior, 3=Manager/Lead, 4=Director/VP/Executive).
  - `Experience_per_Age`: Ratio feature capturing career experience density ($\text{Exp} / \text{Age}$).
  - `Exp_x_Education`: Multiplicative interaction term modeling differential returns to experience by degree tier.
- **Encoding**: Ordinal mapping for Education, binary dummy for Gender (`Gender_Male`), and one-hot encoding for `Job Category`.

### Stage 3: Exploratory Data Analysis & Visualizations (`stages/stage3_eda.py`)
Generates 9 analytical plots in `outputs/plots/`:
1. `01_salary_distribution.png`: Target salary histogram with KDE.
2. `02_salary_boxplot.png`: Univariate salary dispersion and outlier assessment.
3. `03_age_experience_distribution.png`: Dual histograms for workforce age and career tenure.
4. `04_experience_vs_salary.png`: Bivariate scatter showing strong linear correlation ($r = +0.82$).
5. `05_age_vs_salary.png`: Age vs. compensation progression curve.
6. `06_education_vs_salary.png`: Boxplot showing monotonic salary growth across degree levels.
7. `07_job_category_vs_salary.png`: Comparative distributions across the 9 functional job categories.
8. `08_seniority_vs_salary.png`: Boxplot validating salary tiering across seniority levels 0 to 4.
9. `09_gender_vs_salary.png`: Gender salary distribution comparison.

### Stage 4: Statistical Inference & Feature Selection (`stages/stage4_stats.py`)
- **Correlation Heatmap (`10_correlation_heatmap.png`)**: Computes Pearson correlation matrix; analyzes collinearity between Age and Experience ($r \approx 0.98$).
- **Welch's $t$-test (Education)**: Bachelor's mean (\$91,211, $n=768$) vs. Master's mean (\$127,373, $n=567$) yields $t = -15.56, p < 0.0001 \implies$ statistically significant degree premium.
- **One-Way ANOVA**:
  - Across 4 Education levels: $F = 363.42, p < 0.0001$.
  - Across 9 Job Categories: $F = 18.23, p < 0.0001$.
- **Welch's $t$-test (Gender)**: Male vs. Female salary difference evaluation ($t = 4.38, p < 0.0001$).
- **Feature Importance (`11_feature_importance.png`)**: Fits Random Forest to rank predictors; confirms `Years of Experience`, `Age`, and `Seniority` dominate salary variance.

### Stage 5: Regressors, Benchmarking & Inference (`stages/stage5_models.py`)
- Splits data 80/20 train/test (`random_state=42`).
- Scales numerical features with `StandardScaler` (fitted on training data only to avoid leakage).
- Trains and evaluates three candidate regressors:
  1. **Multiple Linear Regression** (OLS)
  2. **Decision Tree Regressor** (`max_depth=8`)
  3. **Random Forest Regressor** (`n_estimators=100`)
- Produces parity plot (`12_actual_vs_predicted.png`) and performance comparison chart (`13_model_r2_comparison.png`).
- Persists best model to `outputs/models/salary_rf_model.joblib`.
- Exposes interactive CLI and single prediction API function `predict_salary()`.

---

## 4. Final Model Comparison & Benchmark

| Regressor Architecture | Train $R^2$ | Test $R^2$ | MAE ($) | RMSE ($) | Overfitting Gap | Best Model? |
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

## 5. Quickstart & Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Entire 5-Stage Pipeline End-to-End
```bash
python main.py
```

### 3. Run Individual Pipeline Stages
```bash
python main.py stage 1    # Data Ingestion & Summary
python main.py stage 2    # Preprocessing, Feature Engineering & Encoding
python main.py stage 3    # Exploratory Data Analysis & Plots 01-09
python main.py stage 4    # Statistical Inference, t-tests, ANOVA & Plot 10-11
python main.py stage 5    # Model Training, Evaluation, Plots 12-13 & Inference
```
*(Each stage can also be executed directly, e.g. `python stages/stage1_load_summary.py`)*

### 4. Interactive Terminal Salary Predictor
```bash
python main.py predict
```
*Allows interactive input of Age, Experience, Gender, Education, and Job Title to receive real-time salary estimates powered by the serialized Random Forest model.*

### 5. Run Automated Quality Assurance Test Suite
```bash
python test_smoke.py
```
*Executes 45 comprehensive validation checks across data integrity, mappings, plots, model joblib binary, and inference edge cases.*
