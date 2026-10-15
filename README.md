# Employee Salary Prediction

**Data Exploration and Visualization Laboratory (DEVL) Mini-Project**

## 1. Project Overview

This project predicts employee salaries using data exploration, visualization, and
multiple linear regression. It demonstrates a complete data science workflow from
raw CSV data to an interactive salary estimator.

## 2. Problem Statement & Objectives

**Problem:** Given employee attributes (age, experience, gender, education level),
can we build a regression model to estimate salary?

**Objectives:**
- Inspect and clean a real employee salary dataset.
- Perform exploratory data analysis with meaningful visualizations.
- Apply statistical analysis including a hypothesis test.
- Train and evaluate a Multiple Linear Regression model.
- Provide an interactive command-line salary prediction tool.

## 3. Dataset

- **File:** `Salary_Data.csv` (6,704 rows, 6 columns)
- **Source:** The dataset was found in the project workspace. Original source is
  unverified (likely a Kaggle salary dataset).
- **Currency:** Not specified in the dataset. Values appear to be in USD based on
  the range (350 - 250,000), but this cannot be confirmed.

| Column              | Type    | Description                    |
|---------------------|---------|--------------------------------|
| Age                 | float64 | Employee age                   |
| Gender              | string  | Male, Female, Other            |
| Education Level     | string  | High School, Bachelor's, Master's, PhD |
| Job Title           | string  | 193 unique job titles          |
| Years of Experience | float64 | Total years of work experience |
| Salary              | float64 | Annual salary (target)         |

## 4. Technologies Used

| Technology    | Purpose                               |
|---------------|---------------------------------------|
| Python 3.14   | Programming language                  |
| Pandas        | Data loading, inspection, cleaning    |
| NumPy         | Numerical operations                  |
| Matplotlib    | Data visualization                    |
| Seaborn       | Statistical visualizations            |
| Scikit-learn  | Preprocessing, regression, evaluation |
| SciPy         | Hypothesis testing (t-test)           |
| Joblib        | Model persistence (save/load)         |

## 5. Project Structure

```
Miniproject/
├── Salary_Data.csv              # Original dataset (unchanged)
├── EmployeeSalaryPrediction.py  # Original exploratory script (preserved)
├── cleaned_salary_data.csv      # Previously generated file (preserved)
├── main.py                      # Entry point (run this)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── test_smoke.py                # Verification test script
├── src/
│   ├── __init__.py
│   └── salary_prediction.py     # Core module (all functions)
└── outputs/
    ├── salary_model.joblib      # Saved trained model
    ├── model_results.txt        # Evaluation metrics
    └── visualizations/
        ├── 01_salary_distribution.png
        ├── 02_experience_vs_salary.png
        ├── 03_salary_by_education.png
        ├── 04_salary_by_job_title.png
        ├── 05_salary_boxplot.png
        ├── 06_correlation_heatmap.png
        ├── 07_age_distribution.png
        └── 08_actual_vs_predicted.png
```

## 6. Data Cleaning & Preprocessing

| Step | Action | Details |
|------|--------|---------|
| 1 | Standardize Education | Mapped "Bachelor's Degree" -> "Bachelor's", "Master's Degree" -> "Master's", "phD" -> "PhD" |
| 2 | Drop missing target | Removed 5 rows with missing Salary |
| 3 | Remove invalid salary | Removed rows with Salary <= 0 |
| 4 | Remove duplicates | Dropped 4,912 duplicate rows |
| 5 | Remove rare gender | Dropped 7 rows with Gender='Other' (too few for analysis) |
| 6 | Impute missing features | 1 missing Education Level filled with mode |
| 7 | Outlier investigation | IQR method found 0 outliers after dedup; kept all valid records |

**Final cleaned dataset:** 1,780 rows, 6 columns, 0 missing values.

## 7. EDA & Visualizations

Eight visualizations are generated in `outputs/visualizations/`:

1. **Salary Distribution** - Histogram with KDE showing right-skewed salary distribution.
2. **Experience vs Salary** - Scatter plot showing strong positive correlation (+0.82).
3. **Salary by Education** - Box plot showing salary increases with education level.
4. **Salary by Top 10 Jobs** - Horizontal box plot of highest-frequency job titles.
5. **Salary Box Plot** - Overall outlier inspection.
6. **Correlation Heatmap** - Shows relationships between Age, Experience, and Salary.
7. **Age Distribution** - Histogram of employee age distribution.
8. **Actual vs Predicted** - Regression evaluation scatter with reference line.

## 8. Statistical Analysis

**Descriptive Statistics (Salary):**
- Mean: 113,163 | Median: 110,000 | Std Dev: 51,602
- Min: 350 | Max: 250,000

**Correlations with Salary:**
- Years of Experience: +0.82 (strong positive)
- Age: +0.77 (strong positive)

**Hypothesis Test:**
- **Question:** Is there a significant salary difference between Bachelor's and Master's holders?
- **Test:** Welch's two-sample t-test (unequal variance)
- **Result:** p < 0.001 -- REJECT H0. There IS a significant difference.
- Bachelor's mean: 91,211 (n=768) vs. Master's mean: 127,373 (n=567)

## 9. Regression Model

**Algorithm:** Multiple Linear Regression (scikit-learn)

**Features used:** Age, Years of Experience, Gender, Education Level

**Excluded:** Job Title (193 unique values would create excessive sparse features
and likely overfit a linear model; noted as a limitation).

**Preprocessing pipeline (fitted on training data only):**
- Numeric features: Median imputation + Standard scaling
- Categorical features: Mode imputation + One-hot encoding (unknown categories ignored)

**Train/Test split:** 80/20 (random_state=42)

### Evaluation Metrics (Test Set)

| Metric | Value |
|--------|-------|
| MAE (Mean Absolute Error) | 21,245.89 |
| MSE (Mean Squared Error) | 774,391,406.01 |
| RMSE (Root Mean Squared Error) | 27,827.89 |
| R² (Coefficient of Determination) | 0.7311 |

**Interpretation:** R² = 0.73 means the model explains approximately 73% of the
variance in salary using only 4 features. This is moderate performance. The
remaining variance is likely explained by Job Title and other factors not captured
in the dataset. R² is NOT a percentage accuracy.

## 10. Key Findings

1. **Experience is the strongest predictor** of salary (correlation = +0.82).
2. **Higher education correlates with higher salary** - PhD holders earn significantly
   more than Bachelor's holders on average.
3. **There is a statistically significant salary gap** between Bachelor's and Master's
   degree holders (p < 0.001).
4. **The linear model captures 73% of salary variance** using only Age, Experience,
   Gender, and Education Level. Including Job Title could improve this but requires
   different modeling techniques (e.g., target encoding, tree-based models).

## 11. Setup & Execution (Windows + VS Code)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the full pipeline (data inspection -> training -> evaluation)
```bash
python main.py
```

### Run interactive salary prediction
```bash
python main.py predict
```

### Run verification tests
```bash
python test_smoke.py
```

## 12. Interactive Prediction

After running the full pipeline (or if the model is already saved), use:

```bash
python main.py predict
```

You will be prompted to enter:
- **Age** (18-70)
- **Years of Experience** (0-50)
- **Gender** (Male/Female)
- **Education Level** (High School/Bachelor's/Master's/PhD)

The model returns an **estimated salary** based on the trained regression model.
Type `q` or `quit` at any prompt to exit.

**Note:** The predicted amount is an approximate model estimate, not a guaranteed
market salary.

## 13. Syllabus Mapping (DEVL)

| Syllabus Topic | Implementation |
|----------------|---------------|
| Data loading & inspection | `pd.read_csv()`, `.head()`, `.shape`, `.info()`, `.describe()`, `.isnull().sum()` |
| Python fundamentals | Variables, functions, lists, dictionaries, conditions, loops |
| NumPy operations | Array operations, `np.sqrt()`, `np.round()`, `np.isfinite()` |
| Data cleaning | Missing value handling, duplicate removal, standardization |
| Matplotlib visualization | Histograms, scatter plots, reference lines |
| Seaborn visualization | Box plots, heatmaps, KDE plots, styled themes |
| Descriptive statistics | Mean, median, std, min, max, correlation |
| Hypothesis testing | Welch's t-test with interpretation |
| Scikit-learn ML pipeline | `Pipeline`, `ColumnTransformer`, `train_test_split` |
| Linear Regression | `LinearRegression`, `.fit()`, `.predict()` |
| Model evaluation | MAE, MSE, RMSE, R² on held-out test set |
| Data exploration workflow | End-to-end: CSV -> EDA -> Model -> Prediction |

## 14. Limitations & Future Improvements

### Limitations
- **Job Title excluded** from features due to high cardinality (193 unique values).
  This limits model accuracy.
- **Dataset source unverified** -- original provenance cannot be confirmed.
- **Salary currency unspecified** -- assumed USD based on value range.
- **Linear model** may not capture non-linear salary patterns.
- **Dataset has many duplicates** (4,912 of 6,704 rows), reducing effective sample size.

### Possible Improvements
- Use target encoding or feature hashing for Job Title.
- Try tree-based models (Random Forest, Gradient Boosting) for better accuracy.
- Add cross-validation for more robust evaluation.
- Include additional features (location, company size, industry) if available.
- Build a simple web interface for predictions.
