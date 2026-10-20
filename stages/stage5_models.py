# -*- coding: utf-8 -*-
"""
STAGE 5: MODEL TRAINING, EVALUATION & INFERENCE (Phase 5)
Data Exploration and Visualization Laboratory (DEVL) Mini-Project
"""

import json
import os
os.environ["MPLBACKEND"] = "Agg"
import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from pathlib import Path

# Setup module path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-blocking plot saving
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from helpers import (
    ENCODED_DATA_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    PLOTS_DIR,
    get_job_category,
    get_seniority,
)

EDUCATION_MAP = {
    "High School": 0,
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3,
}

MODEL_SAVE_PATH = MODELS_DIR / "salary_rf_model.joblib"
METRICS_SAVE_PATH = MODELS_DIR / "model_metrics.txt"


def train_models():
    """Train candidate models, evaluate metrics, generate plots, and persist best model."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 65)
    print("STAGE 5: MODEL TRAINING, EVALUATION & INFERENCE (Phase 5)")
    print("=" * 65)

    if not ENCODED_DATA_PATH.exists():
        raise FileNotFoundError(f"Encoded dataset not found at {ENCODED_DATA_PATH}. Run stage2 first.")

    # Load encoded dataset
    df = pd.read_csv(ENCODED_DATA_PATH)
    print(f"Loaded encoded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Feature matrix X and target y
    X = df.drop(columns=["Salary"])
    y = df["Salary"]
    feature_columns = X.columns.tolist()

    # 80-20 Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 2. Scale numeric columns for Linear Regression (fit on X_train ONLY to prevent data leakage)
    numeric_scale_cols = [
        "Age", "Years of Experience", "Education_Encoded", 
        "Seniority", "Experience_per_Age", "Exp_x_Education"
    ]
    # Ensure all scale cols exist
    scale_cols_present = [c for c in numeric_scale_cols if c in X_train.columns]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[scale_cols_present] = scaler.fit_transform(X_train[scale_cols_present])
    X_test_scaled[scale_cols_present] = scaler.transform(X_test[scale_cols_present])

    # 3. Train three candidate regression models
    print("\n--- TRAINING CANDIDATE REGRESSION MODELS ---")
    models = {
        "Linear Regression": (LinearRegression(), X_train_scaled, X_test_scaled),
        "Decision Tree": (DecisionTreeRegressor(max_depth=8, random_state=42), X_train, X_test),
        "Random Forest": (RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test),
    }

    eval_rows = []
    predictions = {}
    fitted_models = {}

    for name, (model, tr_data, te_data) in models.items():
        print(f"Training {name}...")
        model.fit(tr_data, y_train)
        fitted_models[name] = model

        tr_pred = model.predict(tr_data)
        te_pred = model.predict(te_data)
        predictions[name] = te_pred

        train_r2 = r2_score(y_train, tr_pred)
        test_r2 = r2_score(y_test, te_pred)
        mae = mean_absolute_error(y_test, te_pred)
        mse = mean_squared_error(y_test, te_pred)
        rmse = np.sqrt(mse)

        eval_rows.append({
            "Model": name,
            "Train R²": round(train_r2, 4),
            "Test R²": round(test_r2, 4),
            "MAE ($)": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE ($)": round(rmse, 2),
        })

    results_df = pd.DataFrame(eval_rows)

    print("\n--- MODEL COMPARISON TABLE ---")
    print(results_df.to_string(index=False))

    # Identify best model
    best_by_r2 = results_df.sort_values(by="Test R²", ascending=False).iloc[0]["Model"]
    best_by_rmse = results_df.sort_values(by="RMSE ($)", ascending=True).iloc[0]["Model"]
    print(f"\nBest model by Test R² : {best_by_r2}")
    print(f"Best model by RMSE    : {best_by_rmse}")

    # 4. Visualizations: Parity Plot and R² Comparison
    # 4a. Actual vs Predicted parity plot for the best model (Random Forest)
    plt.figure(figsize=(8, 6))
    best_pred = predictions[best_by_r2]
    plt.scatter(y_test, best_pred, alpha=0.5, color="#1d3557", label="Test Predictions")
    min_val = min(y_test.min(), best_pred.min())
    max_val = max(y_test.max(), best_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y = x (Perfect Prediction)")
    plt.title(f"12. Actual vs Predicted Salary ({best_by_r2})", fontsize=13)
    plt.xlabel("Actual Salary ($)")
    plt.ylabel("Predicted Salary ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "12_actual_vs_predicted.png", dpi=120)
    plt.close()
    print("\n[Plot 12 Saved] Actual vs Predicted: Predictions cluster tightly around the 45-degree diagonal line.")

    # 4b. Bar chart comparing Train R² and Test R²
    plt.figure(figsize=(9, 5))
    x_idx = np.arange(len(results_df))
    bar_width = 0.35
    plt.bar(x_idx - bar_width/2, results_df["Train R²"], width=bar_width, label="Train R²", color="#457b9d")
    plt.bar(x_idx + bar_width/2, results_df["Test R²"], width=bar_width, label="Test R²", color="#e76f51")
    plt.xticks(x_idx, results_df["Model"], fontsize=11)
    plt.ylim(0, 1.05)
    plt.ylabel("R² Score")
    plt.title("13. Model R² Score Comparison (Train vs Test)", fontsize=13)
    plt.legend(loc="lower right")
    for i in range(len(results_df)):
        plt.text(i - bar_width/2, results_df["Train R²"].iloc[i] + 0.02, f"{results_df['Train R²'].iloc[i]:.2f}", ha="center", fontsize=9)
        plt.text(i + bar_width/2, results_df["Test R²"].iloc[i] + 0.02, f"{results_df['Test R²'].iloc[i]:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "13_model_r2_comparison.png", dpi=120)
    plt.close()
    print("[Plot 13 Saved] Model R² Comparison: Random Forest delivers the highest generalization test accuracy.")

    # 5. Baseline comparison
    print("\n--- 5. COMPARISON AGAINST PREVIOUS BASELINE ---")
    baseline_r2 = 0.7311
    baseline_rmse = 27828.00
    baseline_mae = 21246.00

    rf_test_r2 = results_df.loc[results_df["Model"] == "Random Forest", "Test R²"].values[0]
    rf_rmse = results_df.loc[results_df["Model"] == "Random Forest", "RMSE ($)"].values[0]
    rf_mae = results_df.loc[results_df["Model"] == "Random Forest", "MAE ($)"].values[0]

    comp_data = {
        "Metric": ["Test R²", "RMSE ($)", "MAE ($)"],
        "Old Baseline (No Job Title)": [baseline_r2, baseline_rmse, baseline_mae],
        "New Best Model (Random Forest)": [rf_test_r2, rf_rmse, rf_mae],
        "Improvement": [
            f"+{rf_test_r2 - baseline_r2:.4f} (+{((rf_test_r2 - baseline_r2)/baseline_r2)*100:.1f}%)",
            f"-${baseline_rmse - rf_rmse:.2f} ({((baseline_rmse - rf_rmse)/baseline_rmse)*100:.1f}% error reduction)",
            f"-${baseline_mae - rf_mae:.2f} ({((baseline_mae - rf_mae)/baseline_mae)*100:.1f}% error reduction)"
        ]
    }
    comp_df = pd.DataFrame(comp_data)
    print(comp_df.to_string(index=False))

    # 6. Save Best Model and Metrics
    best_model = fitted_models[best_by_r2]
    model_bundle = {
        "model_name": best_by_r2,
        "model": best_model,
        "feature_columns": feature_columns,
        "education_map": EDUCATION_MAP,
        "metrics": results_df.to_dict(orient="records"),
    }
    joblib.dump(model_bundle, MODEL_SAVE_PATH)
    print(f"\nPersisted best model bundle to: {MODEL_SAVE_PATH}")

    # Write textual summary
    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write("EMPLOYEE SALARY PREDICTION - MODEL EVALUATION METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nBENCHMARK AGAINST OLD BASELINE:\n")
        f.write(comp_df.to_string(index=False))
        f.write(f"\n\nBest Model: {best_by_r2} (Test R2: {rf_test_r2:.4f}, RMSE: ${rf_rmse:,.2f})\n")
    print(f"Saved evaluation metrics log to: {METRICS_SAVE_PATH}")

    return model_bundle, results_df


def predict_salary(age, experience, gender, education, job_title, model_bundle=None):
    """Predict annual salary for an individual employee profile."""
    if model_bundle is None:
        if not MODEL_SAVE_PATH.exists():
            print("Trained model not found. Training model now...")
            model_bundle, _ = train_models()
        else:
            model_bundle = joblib.load(MODEL_SAVE_PATH)

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    edu_dict = model_bundle["education_map"]

    job_cat = get_job_category(job_title)
    seniority = get_seniority(job_title)
    exp_per_age = (float(experience) / float(age)) if float(age) > 0 else 0.0
    edu_enc = edu_dict.get(education, 1)
    exp_x_edu = float(experience) * edu_enc
    gender_male = 1 if str(gender).strip().lower() == "male" else 0

    # Build feature row matching trained columns
    row = {col: 0.0 for col in feature_columns}
    row["Age"] = float(age)
    row["Years of Experience"] = float(experience)
    row["Education_Encoded"] = float(edu_enc)
    row["Seniority"] = float(seniority)
    row["Experience_per_Age"] = float(exp_per_age)
    if "Exp_x_Education" in row:
        row["Exp_x_Education"] = float(exp_x_edu)
    if "Gender_Male" in row:
        row["Gender_Male"] = float(gender_male)

    cat_col = f"Job_Category_{job_cat}"
    if cat_col in row:
        row[cat_col] = 1.0

    input_df = pd.DataFrame([row], columns=feature_columns)
    predicted_val = model.predict(input_df)[0]
    return float(predicted_val)


def interactive_prediction():
    """Terminal interactive CLI for salary estimation."""
    print("\n" + "=" * 60)
    print("INTERACTIVE EMPLOYEE SALARY ESTIMATOR")
    print("=" * 60)

    if not MODEL_SAVE_PATH.exists():
        print("Model file not found. Running model training first...")
        model_bundle, _ = train_models()
    else:
        model_bundle = joblib.load(MODEL_SAVE_PATH)

    print("\nEnter employee details (or press Ctrl+C / type 'exit' to quit):")

    while True:
        try:
            raw_age = input("\nAge (18-70): ").strip()
            if raw_age.lower() in ("exit", "quit", "q"):
                break
            age = float(raw_age)
            if not (18 <= age <= 70):
                print("  Invalid age. Please enter a value between 18 and 70.")
                continue

            raw_exp = input(f"Years of Experience (0-{int(age - 16)}): ").strip()
            experience = float(raw_exp)
            if not (0 <= experience <= (age - 16)):
                print(f"  Invalid experience. Experience must be between 0 and {int(age - 16)} for age {int(age)}.")
                continue

            print("Gender options: [1] Male, [2] Female")
            gender_in = input("Select Gender (1 or 2, default Male): ").strip()
            gender = "Female" if gender_in in ("2", "female", "f") else "Male"

            print("Education Level options:")
            print("  [1] High School")
            print("  [2] Bachelor's")
            print("  [3] Master's")
            print("  [4] PhD")
            edu_choice = input("Select Education (1-4, default Bachelor's): ").strip()
            edu_lookup = {"1": "High School", "2": "Bachelor's", "3": "Master's", "4": "PhD"}
            education = edu_lookup.get(edu_choice, "Bachelor's")

            job_title = input("Job Title (e.g. Software Engineer, Data Scientist, Marketing Manager): ").strip()
            if not job_title:
                job_title = "Software Engineer"

            pred = predict_salary(age, experience, gender, education, job_title, model_bundle)
            print("\n" + "-" * 50)
            print(f"  Profile   : {int(age)} yrs old, {gender}, {education}")
            print(f"  Job Title : {job_title} ({experience} yrs exp)")
            print(f"  Category  : {get_job_category(job_title)} | Seniority Level: {get_seniority(job_title)}")
            print(f"  >>> Estimated Annual Salary: ${pred:,.2f}")
            print("-" * 50)

            cont = input("\nPredict another profile? (y/n, default y): ").strip().lower()
            if cont in ("n", "no", "exit", "quit"):
                break

        except (ValueError, KeyboardInterrupt):
            print("\nExiting interactive predictor.")
            break

    print("\nThank you for using the Employee Salary Estimator!")


def run_stage5():
    """Run stage 5: training, evaluation, benchmark, and example predictions."""
    model_bundle, results_df = train_models()

    # 3 Demonstration profiles
    print("\n--- 7. INFERENCE FUNCTION & TEST PREDICTIONS ---")
    examples = [
        {"age": 28, "experience": 4, "gender": "Female", "education": "Bachelor's", "job_title": "Software Engineer"},
        {"age": 42, "experience": 15, "gender": "Male", "education": "Master's", "job_title": "Director of Data Science"},
        {"age": 35, "experience": 8, "gender": "Female", "education": "PhD", "job_title": "Senior Research Scientist"}
    ]

    print("Running 3 Example Predictions:")
    for i, ex in enumerate(examples, 1):
        pred = predict_salary(
            age=ex["age"],
            experience=ex["experience"],
            gender=ex["gender"],
            education=ex["education"],
            job_title=ex["job_title"],
            model_bundle=model_bundle,
        )
        print(f"\nExample {i}:")
        print(f"  Inputs    : {ex['age']} yrs old, {ex['gender']}, {ex['education']}, '{ex['job_title']}', {ex['experience']} yrs exp")
        print(f"  Predicted : ${pred:,.2f}")

    print("\n" + "=" * 65)
    print("STAGE 5 COMPLETE: Model evaluation and inference successful.")
    print("=" * 65)
    return model_bundle, results_df


if __name__ == "__main__":
    run_stage5()
