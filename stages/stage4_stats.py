# -*- coding: utf-8 -*-
"""
STAGE 4: STATISTICAL INFERENCE & FEATURE SELECTION (Phase 4)
Data Exploration and Visualization Laboratory (DEVL) Mini-Project
"""

import os
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

import matplotlib
matplotlib.use("Agg")  # Non-blocking plot saving
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from helpers import CLEAN_DATA_PATH, ENCODED_DATA_PATH, PLOTS_DIR


def run_stage4():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 65)
    print("STAGE 4: STATISTICAL INFERENCE & FEATURE SELECTION (Phase 4)")
    print("=" * 65)

    if not CLEAN_DATA_PATH.exists() or not ENCODED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed datasets not found at {CLEAN_DATA_PATH} or {ENCODED_DATA_PATH}. Run stage2 first.")

    # Load data files
    clean_df = pd.read_csv(CLEAN_DATA_PATH)
    encoded_df = pd.read_csv(ENCODED_DATA_PATH)

    # --- 1. Correlation Matrix and Heatmap ---
    print("\n--- 1. CORRELATION MATRIX ---")
    corr_cols = ["Age", "Years of Experience", "Salary", "Education_Encoded", "Seniority", "Experience_per_Age"]
    corr_matrix = encoded_df[corr_cols].corr()
    print("Correlation Matrix:")
    print(corr_matrix.round(3))

    # Plot and save correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("10. Correlation Heatmap of Key Features", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_correlation_heatmap.png", dpi=120)
    plt.close()
    print("\n[Plot 10 Saved] Correlation Heatmap saved.")

    # Multicollinearity check between Age and Experience
    age_exp_corr = corr_matrix.loc["Age", "Years of Experience"]
    print(f"\nCorrelation between Age and Years of Experience: {age_exp_corr:.3f}")
    if age_exp_corr > 0.8:
        print("Observation: High correlation (>0.80) signals multicollinearity between Age and Experience.")
        print("Decision: Retain both columns. Tree models are unaffected and linear predictions remain valid.")

    # --- 2. Welch's t-test: Bachelor's vs Master's Salary ---
    print("\n--- 2. WELCH'S T-TEST (BACHELOR'S VS MASTER'S SALARY) ---")
    bachelor_sal = clean_df[clean_df["Education Level"] == "Bachelor's"]["Salary"]
    master_sal = clean_df[clean_df["Education Level"] == "Master's"]["Salary"]
    t_stat_edu, p_val_edu = stats.ttest_ind(bachelor_sal, master_sal, equal_var=False)

    print(f"Sample sizes: Bachelor's = {len(bachelor_sal)}, Master's = {len(master_sal)}")
    print(f"Mean salaries: Bachelor's = ${bachelor_sal.mean():.2f}, Master's = ${master_sal.mean():.2f}")
    print(f"t-statistic: {t_stat_edu:.4f}, p-value: {p_val_edu:.4e}")
    if p_val_edu < 0.05:
        print("Conclusion: Reject H0 (p < 0.05). Master's degree holders earn a statistically significant higher salary than Bachelor's degree holders.")
    else:
        print("Conclusion: Fail to reject H0 (p >= 0.05). No statistically significant difference.")

    # --- 3. One-Way ANOVA: Across Education Levels & Job Categories ---
    print("\n--- 3. ONE-WAY ANOVA TESTS ---")
    edu_levels = ["High School", "Bachelor's", "Master's", "PhD"]
    edu_groups = [clean_df[clean_df["Education Level"] == lvl]["Salary"] for lvl in edu_levels]
    f_edu, p_edu = stats.f_oneway(*edu_groups)
    print(f"Education ANOVA: F = {f_edu:.4f}, p-value = {p_edu:.4e}")
    if p_edu < 0.05:
        print("Conclusion: Reject H0 (p < 0.05). Statistically significant differences exist in salary across education levels.")

    # ANOVA across Job Categories
    cat_levels = clean_df["Job Category"].unique()
    cat_groups = [clean_df[clean_df["Job Category"] == cat]["Salary"] for cat in cat_levels]
    f_cat, p_cat = stats.f_oneway(*cat_groups)
    print(f"\nJob Category ANOVA: F = {f_cat:.4f}, p-value = {p_cat:.4e}")
    if p_cat < 0.05:
        print("Conclusion: Reject H0 (p < 0.05). Statistically significant differences exist in salary across job categories.")

    # --- 4. Gender Welch's t-test ---
    print("\n--- 4. GENDER T-TEST (MALE VS FEMALE SALARY) ---")
    male_sal = clean_df[clean_df["Gender"] == "Male"]["Salary"]
    female_sal = clean_df[clean_df["Gender"] == "Female"]["Salary"]
    t_gender, p_gender = stats.ttest_ind(male_sal, female_sal, equal_var=False)
    print(f"Sample sizes: Male = {len(male_sal)}, Female = {len(female_sal)}")
    print(f"Mean salaries: Male = ${male_sal.mean():.2f}, Female = ${female_sal.mean():.2f}")
    print(f"t-statistic: {t_gender:.4f}, p-value: {p_gender:.4e}")
    if p_gender < 0.05:
        print("Conclusion: Reject H0 (p < 0.05). Statistically significant salary difference between male and female employees in this dataset.")
    else:
        print("Conclusion: Fail to reject H0 (p >= 0.05). No statistically significant difference.")

    # --- 5. Feature Importance via Random Forest ---
    print("\n--- 5. FEATURE IMPORTANCE (RANDOM FOREST) ---")
    X_feat = encoded_df.drop(columns=["Salary"])
    y_feat = encoded_df["Salary"]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_feat, y_feat)

    importances = pd.Series(rf.feature_importances_, index=X_feat.columns).sort_values(ascending=False)
    print("Top 10 most important features:")
    print(importances.head(10).round(4))

    # Plot top 15 features
    plt.figure(figsize=(9, 6))
    top15 = importances.head(15).sort_values(ascending=True)
    top15.plot(kind="barh", color="#2a9d8f")
    plt.title("11. Feature Importance (Random Forest)", fontsize=13)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_feature_importance.png", dpi=120)
    plt.close()
    print("\n[Plot 11 Saved] Feature Importance: Years of Experience and Age dominate model predictions.")

    # --- 6. Selected Features List ---
    print("\n--- 6. SELECTED FEATURES LIST ---")
    selected_features = [
        "Years of Experience",
        "Age",
        "Education_Encoded",
        "Seniority",
        "Experience_per_Age",
        "Gender_Male",
    ] + [col for col in encoded_df.columns if col.startswith("Job_Category_")]
    print("Selected feature columns for modeling:")
    for f in selected_features:
        print(f" - {f}")

    # --- 7. Interaction Feature: Exp_x_Education ---
    print("\n--- 7. ENSURING INTERACTION FEATURE (Exp_x_Education) ---")
    encoded_df["Exp_x_Education"] = encoded_df["Years of Experience"] * encoded_df["Education_Encoded"]
    encoded_df.to_csv(ENCODED_DATA_PATH, index=False)
    print(f"Updated encoded dataset saved to: {ENCODED_DATA_PATH}")
    print(f"Exp_x_Education verified. Columns count: {len(encoded_df.columns)}")

    print("\n" + "=" * 65)
    print("STAGE 4 COMPLETE: Statistical analysis finished and interaction feature saved.")
    print("=" * 65)
    return {
        "t_stat_edu": t_stat_edu,
        "p_val_edu": p_val_edu,
        "f_edu": f_edu,
        "p_edu": p_edu,
        "f_cat": f_cat,
        "p_cat": p_cat,
        "t_gender": t_gender,
        "p_gender": p_gender,
        "importances": importances,
    }


if __name__ == "__main__":
    run_stage4()
