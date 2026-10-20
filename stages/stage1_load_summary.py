# -*- coding: utf-8 -*-
"""
STAGE 1: DATASET INGESTION, SCHEMA INSPECTION & NULL CHECK (Phase 1)
Data Exploration and Visualization Laboratory (DEVL) Mini-Project
"""

import sys
from pathlib import Path

# Setup module path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import pandas as pd
from helpers import get_raw_data_path


def run_stage1():
    print("=" * 65)
    print("STAGE 1: DATASET LOADING AND SUMMARY (Phase 1)")
    print("=" * 65)

    data_path = get_raw_data_path()
    if not data_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at expected path: {data_path}")

    # Load raw CSV dataset
    df = pd.read_csv(data_path)
    print(f"\nLoaded dataset from: {data_path.name}")

    # 1. Dataset Shape
    print("\n--- 1. DATASET SHAPE ---")
    print(f"Total Rows   : {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")

    # 2. List of Columns
    print("\n--- 2. COLUMNS ---")
    print(df.columns.tolist())

    # 3. Data Types
    print("\n--- 3. DATA TYPES ---")
    print(df.dtypes)

    # 4. Missing Values per Column
    print("\n--- 4. MISSING VALUES PER COLUMN ---")
    print(df.isnull().sum())

    # 5. Descriptive Statistics (Numeric + Categorical)
    print("\n--- 5. DESCRIPTIVE STATISTICS (ALL COLUMNS) ---")
    print(df.describe(include="all"))

    # 6. Duplicate Rows
    print("\n--- 6. DUPLICATE ROWS ---")
    num_duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {num_duplicates} (out of {len(df)} total rows)")

    # 7. Unique Values in Categorical Columns
    print("\n--- 7. UNIQUE VALUES IN CATEGORICAL COLUMNS ---")
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        n_unique = df[col].nunique(dropna=False)
        print(f"Column '{col}' has {n_unique} unique values (including nulls)")

    # 8. Education Level & Job Titles Inconsistencies
    print("\n--- 8. SPECIFIC CATEGORICAL INSPECTION ---")
    print("Unique Education Level values:")
    print(df["Education Level"].dropna().unique().tolist())
    print(f"\nTotal unique Job Titles count: {df['Job Title'].nunique()}")

    print("\n" + "=" * 65)
    print("STAGE 1 COMPLETE: Summary finished. No output files generated.")
    print("=" * 65)
    return df


if __name__ == "__main__":
    run_stage1()
