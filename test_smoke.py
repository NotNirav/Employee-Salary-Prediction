# -*- coding: utf-8 -*-
"""
Smoke test suite for Employee Salary Prediction (DEVL Mini-Project).
Verifies data pipeline integrity, feature transformations, statistical charts,
model serialization, and inference functionality across all 5 stages.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from stages.helpers import (
    CLEAN_DATA_PATH,
    ENCODED_DATA_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    PLOTS_DIR,
    RAW_DATA_PATH,
    get_job_category,
    get_seniority,
)
from stages.stage5_models import MODEL_SAVE_PATH, METRICS_SAVE_PATH, predict_salary

passed = 0
failed = 0


def check(name: str, condition: bool):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        failed += 1


print("=" * 65)
print("SMOKE TEST: DEVL EMPLOYEE SALARY PREDICTION (5 STAGES)")
print("=" * 65)

# --- 1. Raw Dataset Integrity ---
print("\n--- 1. Raw Dataset Ingestion ---")
try:
    df_raw = pd.read_csv(RAW_DATA_PATH)
    check("Raw dataset file exists at data/raw/Salary_Data.csv", RAW_DATA_PATH.exists())
    check("Raw dataset has rows (N > 0)", len(df_raw) > 0)
    check("Raw dataset has all 6 original columns", df_raw.shape[1] == 6)
    check("Target 'Salary' column exists and is numeric", "Salary" in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw["Salary"]))
except Exception as e:
    check(f"Raw dataset load failed: {e}", False)

# --- 2. Clean Processed Dataset Integrity ---
print("\n--- 2. Clean Processed Dataset (Stage 2) ---")
try:
    df_clean = pd.read_csv(CLEAN_DATA_PATH)
    check("Clean CSV exists at data/processed/clean_salary_data.csv", CLEAN_DATA_PATH.exists())
    check("Clean dataset has 1,780 rows post-deduplication", len(df_clean) == 1780)
    check("Clean dataset has zero missing values", df_clean.isnull().sum().sum() == 0)
    check("No non-positive salaries remain", (df_clean["Salary"] <= 0).sum() == 0)
    check("Education Level strings standardized", set(df_clean["Education Level"].unique()).issubset({"High School", "Bachelor's", "Master's", "PhD"}))
    check("Gender does not contain 'Other'", (df_clean["Gender"] == "Other").sum() == 0)
except Exception as e:
    check(f"Clean dataset validation failed: {e}", False)

# --- 3. Encoded Dataset Integrity ---
print("\n--- 3. Encoded Dataset for Modeling ---")
try:
    df_enc = pd.read_csv(ENCODED_DATA_PATH)
    check("Encoded CSV exists at data/processed/encoded_salary_data.csv", ENCODED_DATA_PATH.exists())
    check("Encoded dataset has matching row count (1,780)", len(df_enc) == 1780)
    check("All encoded columns are numeric", all(pd.api.types.is_numeric_dtype(df_enc[c]) for c in df_enc.columns))
    check("Engineered 'Exp_x_Education' column present", "Exp_x_Education" in df_enc.columns)
    check("Engineered 'Seniority' column present", "Seniority" in df_enc.columns)
    check("Engineered 'Gender_Male' dummy column present", "Gender_Male" in df_enc.columns)
except Exception as e:
    check(f"Encoded dataset validation failed: {e}", False)

# --- 4. Helper Mapping Logic ---
print("\n--- 4. Feature Engineering Helpers ---")
try:
    check("Role 'Software Engineer' maps to 'Engineering/Software'", get_job_category("Software Engineer") == "Engineering/Software")
    check("Role 'Data Scientist' maps to 'Data/Analytics'", get_job_category("Data Scientist") == "Data/Analytics")
    check("Role 'HR Manager' maps to 'HR'", get_job_category("HR Manager") == "HR")
    check("Role 'Financial Analyst' maps to 'Data/Analytics'", get_job_category("Financial Analyst") in ("Data/Analytics", "Finance/Accounting"))
    check("Seniority 'Junior Developer' maps to level 0", get_seniority("Junior Developer") == 0)
    check("Seniority 'Senior Engineer' maps to level 2", get_seniority("Senior Engineer") == 2)
    check("Seniority 'Director of Operations' maps to level 4", get_seniority("Director of Operations") == 4)
except Exception as e:
    check(f"Helper functions failed: {e}", False)

# --- 5. Analytical Visualizations (01 to 13) ---
print("\n--- 5. Visualization Artifacts (Stages 3, 4, 5) ---")
expected_plots = [
    "01_salary_distribution.png",
    "02_salary_boxplot.png",
    "03_age_experience_distribution.png",
    "04_experience_vs_salary.png",
    "05_age_vs_salary.png",
    "06_education_vs_salary.png",
    "07_job_category_vs_salary.png",
    "08_seniority_vs_salary.png",
    "09_gender_vs_salary.png",
    "10_correlation_heatmap.png",
    "11_feature_importance.png",
    "12_actual_vs_predicted.png",
    "13_model_r2_comparison.png",
]
for plot_file in expected_plots:
    p = PLOTS_DIR / plot_file
    check(f"Plot '{plot_file}' exists and non-empty", p.exists() and p.stat().st_size > 5000)

# --- 6. Model Serialization & Deserialization ---
print("\n--- 6. Model Persistence (Stage 5) ---")
try:
    check("Best model binary exists at outputs/models/salary_rf_model.joblib", MODEL_SAVE_PATH.exists())
    bundle = joblib.load(MODEL_SAVE_PATH)
    check("Bundle contains trained model", "model" in bundle and bundle["model"] is not None)
    check("Bundle contains feature column order", "feature_columns" in bundle and len(bundle["feature_columns"]) > 0)
    check("Bundle contains education mapping", "education_map" in bundle and len(bundle["education_map"]) == 4)
    check("Metrics text file exists at outputs/models/model_metrics.txt", METRICS_SAVE_PATH.exists() and METRICS_SAVE_PATH.stat().st_size > 100)
except Exception as e:
    check(f"Model persistence verification failed: {e}", False)

# --- 7. Forward Inference Gateway ---
print("\n--- 7. Inference Gateway & Edge Cases ---")
try:
    # Test case 1: Standard profile
    pred1 = predict_salary(age=30, experience=6, gender="Male", education="Bachelor's", job_title="Software Engineer")
    check("Inference test 1: Valid positive salary", isinstance(pred1, (int, float)) and 40000 < pred1 < 160000)

    # Test case 2: Executive profile
    pred2 = predict_salary(age=45, experience=18, gender="Female", education="Master's", job_title="Director of Data Science")
    check("Inference test 2: Executive higher than mid-level", pred2 > pred1)

    # Test case 3: Unseen / rare job title
    pred3 = predict_salary(age=28, experience=4, gender="Male", education="Bachelor's", job_title="Spacecraft AI Specialist")
    check("Inference test 3: Unseen job title gracefully handled", isinstance(pred3, (int, float)) and pred3 > 0)

    # Test case 4: Entry level high school
    pred4 = predict_salary(age=20, experience=1, gender="Female", education="High School", job_title="Junior Receptionist")
    check("Inference test 4: Entry level valid estimation", isinstance(pred4, (int, float)) and pred4 > 0 and pred4 < pred1)

    # Test case 5: Explicit seniority override
    pred_jr = predict_salary(age=30, experience=6, gender="Male", education="Bachelor's", job_title="Software Engineer", seniority=0)
    pred_sr = predict_salary(age=30, experience=6, gender="Male", education="Bachelor's", job_title="Software Engineer", seniority=2)
    check("Inference test 5: Explicit seniority parameter works (Senior > Junior)", pred_sr > pred_jr)

    # Test case 6: Multi-model prediction and scenario explanation
    multi_res = predict_salary(age=30, experience=6, gender="Male", education="Bachelor's", job_title="Software Engineer", return_all=True)
    check("Inference test 6: Multi-model output returns all 3 candidate models", len(multi_res["predictions"]) == 3 and "Random Forest" in multi_res["predictions"])
    check("Inference test 7: Multi-model output includes scenario explanation", isinstance(multi_res["reason"], list) and len(multi_res["reason"]) >= 3)
except Exception as e:
    check(f"Inference execution failed: {e}", False)

# --- Final Summary ---
print("\n" + "=" * 65)
print(f"SMOKE TEST SUMMARY: {passed} PASSED, {failed} FAILED")
print("=" * 65)

if failed > 0:
    sys.exit(1)