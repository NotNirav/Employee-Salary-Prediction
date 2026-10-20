# -*- coding: utf-8 -*-
"""
STAGE 2: DATA CLEANING, FEATURE CREATION & ENCODING (Phase 2)
Data Exploration and Visualization Laboratory (DEVL) Mini-Project
"""

import os
import sys
from pathlib import Path

# Setup module path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import numpy as np
import pandas as pd
from helpers import (
    CLEAN_DATA_PATH,
    ENCODED_DATA_PATH,
    get_job_category,
    get_raw_data_path,
    get_seniority,
)


def run_stage2():
    print("=" * 65)
    print("STAGE 2: DATA CLEANING, FEATURE CREATION & ENCODING (Phase 2)")
    print("=" * 65)

    data_path = get_raw_data_path()
    if not data_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at expected path: {data_path}")

    # Ensure parent output directories exist
    os.makedirs(CLEAN_DATA_PATH.parent, exist_ok=True)

    # Load raw dataset
    df = pd.read_csv(data_path)
    initial_rows = len(df)
    print(f"Raw dataset loaded: {initial_rows} rows, {df.shape[1]} columns")

    # --- CLEANING STEP 1: Drop missing or invalid Salary ---
    print("\n--- 1. CLEANING TARGET (SALARY) ---")
    missing_sal = df["Salary"].isnull().sum()
    invalid_sal = (df["Salary"] <= 0).sum()
    df = df.dropna(subset=["Salary"])
    df = df[df["Salary"] > 0]
    print(f"Dropped {missing_sal} rows with missing Salary and {invalid_sal} rows with Salary <= 0.")
    print(f"Remaining rows: {len(df)}")

    # --- CLEANING STEP 2: Standardize Education Level strings ---
    print("\n--- 2. STANDARDIZE EDUCATION LEVEL ---")
    edu_map = {
        "Bachelor's Degree": "Bachelor's",
        "Master's Degree": "Master's",
        "phD": "PhD",
    }
    df["Education Level"] = df["Education Level"].replace(edu_map)
    print("Final unique Education Level values:")
    print(df["Education Level"].dropna().unique().tolist())

    # --- CLEANING STEP 3: Drop Duplicate Rows ---
    print("\n--- 3. REMOVE DUPLICATE ROWS ---")
    duplicates_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Duplicate rows removed: {duplicates_count} rows removed.")
    print(f"Remaining rows: {len(df)} (Reason: dataset contains repeated survey scraping entries)")

    # --- CLEANING STEP 4: Drop Gender == 'Other' ---
    print("\n--- 4. FILTER GENDER ---")
    other_gender_count = (df["Gender"] == "Other").sum()
    df = df[df["Gender"] != "Other"]
    print(f"Dropped {other_gender_count} rows where Gender == 'Other' (too few samples for statistical validity)")
    print(f"Remaining rows: {len(df)}")

    # --- CLEANING STEP 5: Impute Remaining Missing Values ---
    print("\n--- 5. MISSING VALUE IMPUTATION ---")
    print("Null values before imputation:")
    print(df.isnull().sum())

    # Median for numeric features
    for col in ["Age", "Years of Experience"]:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Mode for categorical features
    for col in ["Gender", "Education Level", "Job Title"]:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    print("\nNull values after imputation:")
    print(df.isnull().sum())

    # --- CLEANING STEP 6: Outlier Check on Salary ---
    print("\n--- 6. SALARY OUTLIERS (IQR METHOD) ---")
    q1 = df["Salary"].quantile(0.25)
    q3 = df["Salary"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = ((df["Salary"] < lower_bound) | (df["Salary"] > upper_bound)).sum()
    print(f"IQR outlier count: {outlier_count} potential outliers detected (kept as legitimate senior compensation).")
    print(f"Total cleaned rows: {len(df)}")

    # --- FEATURE CREATION STEP 7: Job Category Mapping ---
    print("\n--- 7. JOB CATEGORY FEATURE CREATION ---")
    title_counts = df["Job Title"].value_counts()
    print(f"Total unique Job Titles: {len(title_counts)}")
    print("Sample of Job Titles and counts:")
    print(title_counts.head(10))

    # Apply keyword grouping from helpers
    df["Job Category"] = df["Job Title"].apply(get_job_category)
    print("\nValue counts per Job Category:")
    print(df["Job Category"].value_counts())

    # --- FEATURE CREATION STEP 8: Seniority Level ---
    print("\n--- 8. SENIORITY FEATURE CREATION ---")
    df["Seniority"] = df["Job Title"].apply(get_seniority)
    print("Seniority distribution:")
    print(df["Seniority"].value_counts().sort_index())

    # --- FEATURE CREATION STEP 9: Experience per Age ---
    print("\n--- 9. RATIO FEATURE CREATION ---")
    df["Experience_per_Age"] = np.where(df["Age"] > 0, df["Years of Experience"] / df["Age"], 0.0)
    print("Experience_per_Age summary:")
    print(df["Experience_per_Age"].describe())

    # Save clean human-readable dataset
    clean_csv_path = CLEAN_DATA_PATH
    df.to_csv(clean_csv_path, index=False)
    print(f"\nSaved human-readable data to: {clean_csv_path}")
    print(f"Shape: {df.shape}")

    # --- ENCODING STEP 10: Education Ordinal Encoding ---
    print("\n--- 10. ENCODING CATEGORICAL FEATURES ---")
    edu_dict = {
        "High School": 0,
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3
    }
    df["Education_Encoded"] = df["Education Level"].map(edu_dict)

    # --- INTERACTION STEP 11: Exp_x_Education ---
    df["Exp_x_Education"] = df["Years of Experience"] * df["Education_Encoded"]

    # --- ENCODING STEP 12: Gender One-Hot (drop_first=True) ---
    gender_dummies = pd.get_dummies(df["Gender"], prefix="Gender", drop_first=True, dtype=int)

    # --- ENCODING STEP 13: Job Category One-Hot ---
    job_dummies = pd.get_dummies(df["Job Category"], prefix="Job_Category", dtype=int)

    # Assemble model-ready numeric dataframe
    numeric_base = df[["Age", "Years of Experience", "Education_Encoded", "Seniority", "Experience_per_Age", "Exp_x_Education", "Salary"]]
    encoded_df = pd.concat([numeric_base, gender_dummies, job_dummies], axis=1)

    # Save encoded dataset
    encoded_csv_path = ENCODED_DATA_PATH
    encoded_df.to_csv(encoded_csv_path, index=False)
    print(f"\nSaved fully numeric model-ready data to: {encoded_csv_path}")
    print(f"Shape: {encoded_df.shape}")

    print("\n" + "=" * 65)
    print("STAGE 2 COMPLETE: Preprocessing finished successfully.")
    print("=" * 65)
    return df, encoded_df


if __name__ == "__main__":
    run_stage2()
