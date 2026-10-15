# -*- coding: utf-8 -*-
"""
Employee Salary Prediction -- Core Module
==========================================
Data Exploration, Visualization and Regression (DEVL) Mini-Project.

This module contains all functions for:
  1. Data Inspection
  2. Data Cleaning
  3. Exploratory Data Analysis & Visualization
  4. Statistical Analysis
  5. Regression Model Training & Evaluation
  6. Interactive Salary Prediction
"""

import os
import sys
import warnings
from pathlib import Path

# Force UTF-8 output on Windows to avoid encoding errors with special chars
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Salary_Data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VIZ_DIR = OUTPUT_DIR / "visualizations"
MODEL_PATH = OUTPUT_DIR / "salary_model.joblib"
RESULTS_PATH = OUTPUT_DIR / "model_results.txt"

RANDOM_SEED = 42


def ensure_dirs():
    """Create output directories if they don't exist."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. DATA INSPECTION
# ---------------------------------------------------------------------------
def load_and_inspect(filepath=DATA_PATH):
    """Load the CSV dataset and print inspection details."""
    print("=" * 60)
    print("STEP 1: DATA INSPECTION")
    print("=" * 60)

    df = pd.read_csv(filepath)

    print(f"\nDataset path : {filepath}")
    print(f"Shape        : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nColumn data types:\n{df.dtypes}")
    print(f"\nDataset info:")
    df.info()
    print(f"\nDescriptive statistics:\n{df.describe()}")
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    print(f"\nTotal duplicate rows: {df.duplicated().sum()}")

    # Unique values for categorical columns
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        unique_vals = df[col].dropna().unique()
        print(f"\nUnique '{col}' ({len(unique_vals)}): {list(unique_vals[:20])}")
        if len(unique_vals) > 20:
            print(f"  ... and {len(unique_vals) - 20} more")

    print(f"\nTarget column: 'Salary'")
    print(f"  Non-null count : {df['Salary'].notna().sum()}")
    print(f"  Min: {df['Salary'].min()}, Max: {df['Salary'].max()}")

    return df


# ---------------------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------------------
def clean_data(df):
    """
    Clean the dataset:
    - Standardize Education Level labels
    - Drop rows with missing Salary (target)
    - Drop duplicate rows
    - Remove clearly invalid salaries (<= 0)
    - Remove records with Gender='Other' (too few for meaningful analysis)
    - Impute remaining missing values for features
    - Investigate outliers (document but keep valid high earners)
    """
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING")
    print("=" * 60)

    original_shape = df.shape
    df = df.copy()

    # --- 2a. Standardize Education Level ---
    edu_map = {
        "Bachelor's Degree": "Bachelor's",
        "Master's Degree": "Master's",
        "phD": "PhD",
    }
    df["Education Level"] = df["Education Level"].replace(edu_map)
    print(f"\nStandardized Education Level values: {list(df['Education Level'].dropna().unique())}")

    # --- 2b. Drop rows where target (Salary) is missing ---
    missing_salary = df["Salary"].isnull().sum()
    df = df.dropna(subset=["Salary"])
    print(f"Dropped {missing_salary} rows with missing Salary.")

    # --- 2c. Remove clearly invalid salaries ---
    invalid_salary = (df["Salary"] <= 0).sum()
    df = df[df["Salary"] > 0]
    print(f"Dropped {invalid_salary} rows with Salary <= 0.")

    # --- 2d. Drop duplicate rows ---
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Dropped {dup_count} duplicate rows.")

    # --- 2e. Remove Gender='Other' (very few records) ---
    other_count = (df["Gender"] == "Other").sum()
    df = df[df["Gender"] != "Other"]
    print(f"Dropped {other_count} rows with Gender='Other'.")

    # --- 2f. Handle remaining missing values in features ---
    for col in ["Age", "Years of Experience"]:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed {n_miss} missing '{col}' with median ({median_val}).")

    for col in ["Gender", "Education Level", "Job Title"]:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"Imputed {n_miss} missing '{col}' with mode ('{mode_val}').")

    # --- 2g. Outlier Investigation ---
    q1 = df["Salary"].quantile(0.25)
    q3 = df["Salary"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (df["Salary"] < lower_bound) | (df["Salary"] > upper_bound)
    n_outliers = outlier_mask.sum()
    print(f"\nOutlier investigation (IQR method):")
    print(f"  Q1={q1:.0f}, Q3={q3:.0f}, IQR={iqr:.0f}")
    print(f"  Lower bound={lower_bound:.0f}, Upper bound={upper_bound:.0f}")
    print(f"  Potential outliers: {n_outliers}")
    if n_outliers > 0:
        print(f"  Outlier salary range: {df.loc[outlier_mask, 'Salary'].min():.0f} - {df.loc[outlier_mask, 'Salary'].max():.0f}")
    # Keep outliers -- extreme salaries may be valid senior/executive roles
    print("  Decision: Keeping outliers (may represent valid high/low earners).")

    print(f"\nCleaned shape: {df.shape} (was {original_shape})")
    print(f"Remaining missing values:\n{df.isnull().sum()}")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS & VISUALIZATION
# ---------------------------------------------------------------------------
def run_eda(df):
    """Generate EDA visualizations and save to outputs/visualizations/."""
    print("\n" + "=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    ensure_dirs()
    sns.set_theme(style="whitegrid", palette="muted")

    saved_plots = []

    # --- 3a. Salary distribution histogram ---
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["Salary"], bins=30, kde=True, color="steelblue", ax=ax)
    ax.set_title("Distribution of Employee Salaries", fontsize=14)
    ax.set_xlabel("Salary")
    ax.set_ylabel("Frequency")
    path = VIZ_DIR / "01_salary_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_plots.append(path)

    # --- 3b. Experience vs Salary scatter ---
    if "Years of Experience" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.scatterplot(x="Years of Experience", y="Salary", data=df,
                        alpha=0.5, color="darkorange", ax=ax)
        ax.set_title("Years of Experience vs Salary", fontsize=14)
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Salary")
        path = VIZ_DIR / "02_experience_vs_salary.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_plots.append(path)
    else:
        print("  Skipped: 'Years of Experience' column not found.")

    # --- 3c. Salary by Education Level ---
    if "Education Level" in df.columns:
        edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
        edu_order = [e for e in edu_order if e in df["Education Level"].values]
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(x="Education Level", y="Salary", data=df,
                    order=edu_order, palette="Set2", ax=ax)
        ax.set_title("Salary by Education Level", fontsize=14)
        ax.set_xlabel("Education Level")
        ax.set_ylabel("Salary")
        path = VIZ_DIR / "03_salary_by_education.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_plots.append(path)
    else:
        print("  Skipped: 'Education Level' column not found.")

    # --- 3d. Salary by Top 10 Job Titles ---
    if "Job Title" in df.columns:
        top_jobs = df["Job Title"].value_counts().head(10).index.tolist()
        df_top = df[df["Job Title"].isin(top_jobs)]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x="Salary", y="Job Title", data=df_top,
                    order=top_jobs, palette="coolwarm", ax=ax)
        ax.set_title("Salary by Top 10 Job Titles", fontsize=14)
        ax.set_xlabel("Salary")
        ax.set_ylabel("Job Title")
        path = VIZ_DIR / "04_salary_by_job_title.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_plots.append(path)
    else:
        print("  Skipped: 'Job Title' column not found.")

    # --- 3e. Salary box plot (outlier inspection) ---
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(x=df["Salary"], color="lightcoral", ax=ax)
    ax.set_title("Salary Box Plot (Outlier Inspection)", fontsize=14)
    ax.set_xlabel("Salary")
    path = VIZ_DIR / "05_salary_boxplot.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_plots.append(path)

    # --- 3f. Correlation heatmap ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrRd",
                    linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap (Numerical Features)", fontsize=14)
        path = VIZ_DIR / "06_correlation_heatmap.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_plots.append(path)
    else:
        print("  Skipped: Not enough numerical columns for correlation heatmap.")

    # --- 3g. Age distribution ---
    if "Age" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(df["Age"], bins=25, kde=True, color="mediumseagreen", ax=ax)
        ax.set_title("Age Distribution of Employees", fontsize=14)
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        path = VIZ_DIR / "07_age_distribution.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_plots.append(path)

    # Verify plots
    for p in saved_plots:
        size = p.stat().st_size
        status = "OK" if size > 0 else "EMPTY!"
        print(f"  Saved: {p.name} ({size:,} bytes) [{status}]")

    return saved_plots


# ---------------------------------------------------------------------------
# 4. STATISTICAL ANALYSIS
# ---------------------------------------------------------------------------
def statistical_analysis(df):
    """Descriptive stats, correlation, and a hypothesis test."""
    print("\n" + "=" * 60)
    print("STEP 4: STATISTICAL ANALYSIS")
    print("=" * 60)

    # Descriptive statistics for Salary
    salary = df["Salary"]
    desc = {
        "Count": len(salary),
        "Mean": salary.mean(),
        "Median": salary.median(),
        "Std Dev": salary.std(),
        "Min": salary.min(),
        "Max": salary.max(),
    }
    print("\nSalary Descriptive Statistics:")
    for k, v in desc.items():
        print(f"  {k:10s}: {v:>12,.2f}")

    # Correlation of numerical features with Salary
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Salary" in num_cols:
        corr_with_salary = df[num_cols].corr()["Salary"].drop("Salary").sort_values(ascending=False)
        print("\nCorrelation with Salary:")
        for feat, val in corr_with_salary.items():
            print(f"  {feat:25s}: {val:+.4f}")

    # Hypothesis test: Is there a significant salary difference between
    # employees with a Bachelor's and those with a Master's degree?
    if "Education Level" in df.columns:
        bachelors = df.loc[df["Education Level"] == "Bachelor's", "Salary"]
        masters = df.loc[df["Education Level"] == "Master's", "Salary"]
        if len(bachelors) >= 20 and len(masters) >= 20:
            print("\n--- Hypothesis Test ---")
            print("Question: Is there a statistically significant difference")
            print("          in mean salary between Bachelor's and Master's holders?")
            print(f"  H0: mean_Bachelor = mean_Master  (no difference)")
            print(f"  H1: mean_Bachelor != mean_Master (significant difference)")
            t_stat, p_value = stats.ttest_ind(bachelors, masters, equal_var=False)
            alpha = 0.05
            print(f"  Test: Welch's two-sample t-test (unequal variance)")
            print(f"  t-statistic : {t_stat:.4f}")
            print(f"  p-value     : {p_value:.6f}")
            print(f"  Significance: alpha = {alpha}")
            if p_value < alpha:
                print(f"  Result: REJECT H0 -- There IS a statistically significant")
                print(f"          salary difference (p < {alpha}).")
            else:
                print(f"  Result: FAIL TO REJECT H0 -- No statistically significant")
                print(f"          salary difference at alpha = {alpha}.")
            print(f"  Bachelor's mean: {bachelors.mean():,.0f} (n={len(bachelors)})")
            print(f"  Master's mean  : {masters.mean():,.0f} (n={len(masters)})")
        else:
            print("\n  Skipped hypothesis test: insufficient samples per group.")
    else:
        print("\n  Skipped hypothesis test: 'Education Level' column not found.")

    return desc


# ---------------------------------------------------------------------------
# 5. REGRESSION MODEL
# ---------------------------------------------------------------------------

# Features used for training
NUMERIC_FEATURES = ["Age", "Years of Experience"]
CATEGORICAL_FEATURES = ["Gender", "Education Level"]
# Note: 'Job Title' has 190+ unique values. Including it raw would
# create a very sparse matrix and overfit. We exclude it for the
# baseline linear regression model and note this as a limitation.
TARGET = "Salary"


def build_pipeline():
    """Build a scikit-learn preprocessing + regression pipeline."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    return pipeline


def train_and_evaluate(df):
    """
    Train a Linear Regression model and evaluate on a held-out test set.
    Returns the pipeline, metrics dict, and test data for further use.
    """
    print("\n" + "=" * 60)
    print("STEP 5: REGRESSION MODEL TRAINING & EVALUATION")
    print("=" * 60)

    ensure_dirs()

    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    # Verify columns exist
    missing_cols = [c for c in all_features + [TARGET] if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return None, None, None, None

    X = df[all_features]
    y = df[TARGET]

    print(f"\nFeatures used : {all_features}")
    print(f"Target        : {TARGET}")
    print(f"Excluded      : Job Title (193 unique values -- too sparse for linear model)")
    print(f"Dataset size  : {len(df)} samples")

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"Training set  : {len(X_train)} samples")
    print(f"Test set      : {len(X_test)} samples")

    # Build and train pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("Model trained successfully.")

    # Predictions on test set
    y_pred = pipeline.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    print("\n--- Model Evaluation Metrics ---")
    print(f"  MAE  (Mean Absolute Error)         : {mae:>12,.2f}")
    print(f"  MSE  (Mean Squared Error)          : {mse:>15,.2f}")
    print(f"  RMSE (Root Mean Squared Error)     : {rmse:>12,.2f}")
    print(f"  R2   (Coefficient of Determination): {r2:>8.4f}")

    # Interpret R2
    if r2 < 0:
        print("  WARNING: R2 is negative -- the model performs worse than predicting the mean.")
    elif r2 < 0.5:
        print("  R2 is low -- the model explains less than 50% of salary variance.")
    elif r2 < 0.8:
        print("  R2 is moderate -- the model captures some salary patterns.")
    else:
        print("  R2 is good -- the model explains most of the salary variance.")

    # --- Actual vs Predicted scatter ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, color="teal", edgecolors="k", linewidths=0.3)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal (y=x)")
    ax.set_title("Actual vs Predicted Salary", fontsize=14)
    ax.set_xlabel("Actual Salary")
    ax.set_ylabel("Predicted Salary")
    ax.legend()
    path = VIZ_DIR / "08_actual_vs_predicted.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {path.name} ({path.stat().st_size:,} bytes)")

    # --- Comparison table ---
    comparison = pd.DataFrame({
        "Actual Salary": y_test.values[:10],
        "Predicted Salary": np.round(y_pred[:10], 2),
        "Error": np.round(y_test.values[:10] - y_pred[:10], 2),
    })
    print("\n--- Sample Predictions (first 10 test records) ---")
    print(comparison.to_string(index=False))

    # --- Save model ---
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # --- Save results to text file ---
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("Employee Salary Prediction -- Model Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Features: {all_features}\n")
        f.write(f"Target: {TARGET}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        f.write("Evaluation Metrics:\n")
        for name, val in metrics.items():
            f.write(f"  {name}: {val:,.4f}\n")
        f.write(f"\nSample Predictions:\n{comparison.to_string(index=False)}\n")
    print(f"Results saved to: {RESULTS_PATH}")

    return pipeline, metrics, X_test, y_test


# ---------------------------------------------------------------------------
# 6. MODEL PERSISTENCE VERIFICATION
# ---------------------------------------------------------------------------
def verify_saved_model():
    """Load the saved model and make a test prediction."""
    print("\n--- Model Persistence Verification ---")
    if not MODEL_PATH.exists():
        print("  Model file not found. Skipping verification.")
        return False
    loaded = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{
        "Age": 30,
        "Years of Experience": 5,
        "Gender": "Male",
        "Education Level": "Bachelor's",
    }])
    pred = loaded.predict(sample)[0]
    print(f"  Loaded model from: {MODEL_PATH}")
    print(f"  Test prediction (Age=30, Exp=5, Male, Bachelor's): {pred:,.2f}")
    return True


# ---------------------------------------------------------------------------
# 7. INTERACTIVE PREDICTION
# ---------------------------------------------------------------------------
VALID_GENDERS = ["Male", "Female"]
VALID_EDUCATION = ["High School", "Bachelor's", "Master's", "PhD"]


def get_numeric_input(prompt, min_val=0, max_val=None):
    """Prompt user for a numeric value with validation."""
    while True:
        raw = input(prompt).strip()
        if raw.lower() in ("q", "quit", "exit"):
            return None
        try:
            val = float(raw)
            if val < min_val:
                print(f"  Value must be >= {min_val}. Try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"  Value must be <= {max_val}. Try again.")
                continue
            return val
        except ValueError:
            print("  Invalid number. Please enter a numeric value.")


def get_choice_input(prompt, choices):
    """Prompt user to pick from a list of valid options."""
    choices_lower = {c.lower(): c for c in choices}
    while True:
        raw = input(prompt).strip()
        if raw.lower() in ("q", "quit", "exit"):
            return None
        if raw.lower() in choices_lower:
            return choices_lower[raw.lower()]
        # Try exact match
        if raw in choices:
            return raw
        print(f"  Invalid choice. Options: {choices}")


def interactive_prediction(pipeline=None):
    """
    Command-line interface to predict salary based on user input.
    Uses the same preprocessing pipeline from training.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE SALARY PREDICTION")
    print("=" * 60)

    # Load model if not provided
    if pipeline is None:
        if not MODEL_PATH.exists():
            print("ERROR: Trained model not found. Run the training first.")
            return
        pipeline = joblib.load(MODEL_PATH)
        print(f"Loaded model from: {MODEL_PATH}")

    print("\nEnter employee details to get a salary estimate.")
    print("Type 'q' or 'quit' at any prompt to exit.\n")
    print(f"Accepted features:")
    print(f"  - Age (numeric, 18-70)")
    print(f"  - Years of Experience (numeric, 0-50)")
    print(f"  - Gender ({', '.join(VALID_GENDERS)})")
    print(f"  - Education Level ({', '.join(VALID_EDUCATION)})")
    print()

    while True:
        print("-" * 40)
        age = get_numeric_input("Enter Age (18-70): ", min_val=18, max_val=70)
        if age is None:
            break

        exp = get_numeric_input("Enter Years of Experience (0-50): ", min_val=0, max_val=50)
        if exp is None:
            break

        # Validate: experience shouldn't exceed (age - 16)
        if exp > age - 16:
            print(f"  Warning: {exp} years of experience seems high for age {age:.0f}.")
            print("  Proceeding with the given values.\n")

        gender = get_choice_input(f"Enter Gender ({'/'.join(VALID_GENDERS)}): ", VALID_GENDERS)
        if gender is None:
            break

        education = get_choice_input(
            f"Enter Education Level ({'/'.join(VALID_EDUCATION)}): ",
            VALID_EDUCATION,
        )
        if education is None:
            break

        # Build input DataFrame
        input_data = pd.DataFrame([{
            "Age": age,
            "Years of Experience": exp,
            "Gender": gender,
            "Education Level": education,
        }])

        # Predict
        predicted_salary = pipeline.predict(input_data)[0]

        print()
        print("  +--------------------------------------+")
        print("  |  Input Summary                       |")
        print(f"  |  Age              : {age:<17.0f}|")
        print(f"  |  Experience       : {exp:<17.0f}|")
        print(f"  |  Gender           : {gender:<17s}|")
        print(f"  |  Education Level  : {education:<17s}|")
        print("  +--------------------------------------+")
        print(f"  |  Estimated Salary : {predicted_salary:>14,.2f}   |")
        print("  +--------------------------------------+")
        print("  Note: This is an approximate model estimate, not a guaranteed salary.\n")

    print("\nExiting prediction interface. Goodbye!")


# ---------------------------------------------------------------------------
# COMPLETE WORKFLOW (non-interactive)
# ---------------------------------------------------------------------------
def run_full_pipeline():
    """Run the complete data science workflow (Steps 1-6)."""
    df = load_and_inspect()
    df_clean = clean_data(df)
    run_eda(df_clean)
    statistical_analysis(df_clean)
    pipeline, metrics, X_test, y_test = train_and_evaluate(df_clean)
    if pipeline is not None:
        verify_saved_model()
    return pipeline, metrics
