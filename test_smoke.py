# -*- coding: utf-8 -*-
"""
Smoke test for Employee Salary Prediction project.
Verifies all core functionality works correctly.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib

from src.salary_prediction import (
    DATA_PATH, MODEL_PATH, VIZ_DIR, RESULTS_PATH,
    clean_data, train_and_evaluate,
)

passed = 0
failed = 0
df = None
df_clean = None
pipeline = None


def check(name, condition):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        failed += 1


print("=" * 60)
print("SMOKE TEST: Employee Salary Prediction")
print("=" * 60)

# 1. Dataset loads
print("\n--- 1. Dataset Loading ---")
try:
    df = pd.read_csv(DATA_PATH)
    check("Dataset loads successfully", True)
    check("Dataset has rows", df.shape[0] > 0)
    check("Dataset has 6 columns", df.shape[1] == 6)
    check("Salary column exists", "Salary" in df.columns)
    check("Salary has numeric values", pd.api.types.is_numeric_dtype(df["Salary"]))
except Exception as e:
    check(f"Dataset loading failed: {e}", False)

# 2. Column mapping
print("\n--- 2. Column Identification ---")
if df is not None:
    expected_cols = ["Age", "Gender", "Education Level", "Job Title",
                     "Years of Experience", "Salary"]
    for col in expected_cols:
        check(f"Column '{col}' exists", col in df.columns)
else:
    check("Cannot check columns - dataset not loaded", False)

# 3. Data cleaning
print("\n--- 3. Data Cleaning ---")
if df is not None:
    try:
        df_clean = clean_data(df)
        check("Cleaning completes without error", True)
        check("No missing values after cleaning", df_clean.isnull().sum().sum() == 0)
        check("No duplicates after cleaning", df_clean.duplicated().sum() == 0)
        check("All salaries > 0", (df_clean["Salary"] > 0).all())
        check("Cleaned dataset has rows", len(df_clean) > 100)
    except Exception as e:
        check(f"Data cleaning failed: {e}", False)
else:
    check("Cannot clean - dataset not loaded", False)

# 4. EDA plots
print("\n--- 4. EDA Visualizations ---")
expected_plots = [
    "01_salary_distribution.png",
    "02_experience_vs_salary.png",
    "03_salary_by_education.png",
    "04_salary_by_job_title.png",
    "05_salary_boxplot.png",
    "06_correlation_heatmap.png",
    "07_age_distribution.png",
]
for pname in expected_plots:
    p = VIZ_DIR / pname
    check(f"{pname} exists and non-empty", p.exists() and p.stat().st_size > 0)

# 5. Model training and evaluation
print("\n--- 5. Model Training & Evaluation ---")
if df_clean is not None:
    try:
        pipeline, metrics, X_test, y_test = train_and_evaluate(df_clean)
        check("Pipeline trained successfully", pipeline is not None)
        check("MAE is finite", np.isfinite(metrics["MAE"]))
        check("MSE is finite", np.isfinite(metrics["MSE"]))
        check("RMSE is finite", np.isfinite(metrics["RMSE"]))
        check("R2 is finite", np.isfinite(metrics["R2"]))
        check("R2 > 0 (better than mean predictor)", metrics["R2"] > 0)
    except Exception as e:
        check(f"Model training failed: {e}", False)
else:
    check("Cannot train - cleaned data not available", False)

# 6. Actual vs Predicted plot
print("\n--- 6. Actual vs Predicted Plot ---")
avp = VIZ_DIR / "08_actual_vs_predicted.png"
check("actual_vs_predicted.png exists and non-empty",
      avp.exists() and avp.stat().st_size > 0)

# 7. Model persistence
print("\n--- 7. Model Persistence ---")
check("Model file exists", MODEL_PATH.exists())
try:
    loaded = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{
        "Age": 35, "Years of Experience": 10,
        "Gender": "Female", "Education Level": "Master's",
    }])
    pred = loaded.predict(sample)[0]
    check("Loaded model produces prediction", np.isfinite(pred))
    check("Prediction is positive", pred > 0)
except Exception as e:
    check(f"Model loading failed: {e}", False)

# 8. Prediction with valid input
print("\n--- 8. Prediction Function ---")
if pipeline is not None:
    try:
        sample_input = pd.DataFrame([{
            "Age": 28, "Years of Experience": 3,
            "Gender": "Male", "Education Level": "Bachelor's",
        }])
        pred = pipeline.predict(sample_input)[0]
        check("Valid input returns prediction", np.isfinite(pred) and pred > 0)
    except Exception as e:
        check(f"Prediction function failed: {e}", False)
else:
    check("Cannot predict - pipeline not trained", False)

# 9. Unseen category handling
print("\n--- 9. Unseen Category Handling ---")
if pipeline is not None:
    try:
        unseen_input = pd.DataFrame([{
            "Age": 40, "Years of Experience": 15,
            "Gender": "NonBinary", "Education Level": "Diploma",
        }])
        pred = pipeline.predict(unseen_input)[0]
        check("Unseen categories handled gracefully", np.isfinite(pred))
    except Exception as e:
        check(f"Unseen category handling failed: {e}", False)
else:
    check("Cannot test unseen categories - pipeline not trained", False)

# 10. Results file
print("\n--- 10. Output Files ---")
check("model_results.txt exists", RESULTS_PATH.exists())
check("model_results.txt non-empty",
      RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0)

# 11. Original files preserved
print("\n--- 11. Original Files Preserved ---")
orig_csv = PROJECT_ROOT / "Salary_Data.csv"
orig_py = PROJECT_ROOT / "EmployeeSalaryPrediction.py"
check("Original Salary_Data.csv preserved", orig_csv.exists())
check("Original EmployeeSalaryPrediction.py preserved", orig_py.exists())

# Summary
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 60)
if failed > 0:
    print("Some checks FAILED. Review the output above.")
    sys.exit(1)
else:
    print("All checks PASSED!")
    sys.exit(0)
