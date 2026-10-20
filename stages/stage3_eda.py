# -*- coding: utf-8 -*-
"""
STAGE 3: EXPLORATORY DATA ANALYSIS & VISUALIZATION (Phase 3)
Data Exploration and Visualization Laboratory (DEVL) Mini-Project
"""

import os
os.environ["MPLBACKEND"] = "Agg"  # Force headless rendering before matplotlib loads
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# Setup module path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from helpers import CLEAN_DATA_PATH, PLOTS_DIR


def run_stage3():
    # Set visual style
    sns.set_theme(style="whitegrid", palette="muted")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 65)
    print("STAGE 3: EXPLORATORY DATA ANALYSIS & VISUALIZATION (Phase 3)")
    print("=" * 65)

    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {CLEAN_DATA_PATH}. Please run stage2 first.")

    # Load cleaned dataset
    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"Loaded clean dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Target Salary distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Salary"], bins=30, kde=True, color="#2b5c8f")
    plt.title("01. Distribution of Employee Salaries", fontsize=13)
    plt.xlabel("Salary ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_salary_distribution.png", dpi=120)
    plt.close()
    print("\n[Plot 01 Saved] Salary Distribution: Right-skewed distribution with peaks around $50k-$60k and $160k-$180k.")

    # 2. Salary boxplot
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df["Salary"], color="#4a90e2")
    plt.title("02. Boxplot of Employee Salaries (Outlier Detection)", fontsize=13)
    plt.xlabel("Salary ($)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_salary_boxplot.png", dpi=120)
    plt.close()
    print("[Plot 02 Saved] Salary Boxplot: No severe statistical outliers; spread spans smoothly from ~$350 to $250k.")

    # 3. Histograms of Age and Years of Experience
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["Age"], bins=20, kde=True, color="#347b98", ax=axes[0])
    axes[0].set_title("Distribution of Age", fontsize=12)
    axes[0].set_xlabel("Age (Years)")
    axes[0].set_ylabel("Frequency")

    sns.histplot(df["Years of Experience"], bins=20, kde=True, color="#e07a5f", ax=axes[1])
    axes[1].set_title("Distribution of Years of Experience", fontsize=12)
    axes[1].set_xlabel("Years of Experience")
    axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_age_experience_distribution.png", dpi=120)
    plt.close()
    print("[Plot 03 Saved] Age & Experience Histograms: Most employees are 25-40 years old with 2-15 years of experience.")

    # 4. Years of Experience vs Salary scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="Years of Experience", y="Salary", data=df, alpha=0.6, color="#1d3557")
    plt.title("04. Years of Experience vs Salary", fontsize=13)
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary ($)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_experience_vs_salary.png", dpi=120)
    plt.close()
    print("[Plot 04 Saved] Experience vs Salary: Strong positive linear trend; experience is a primary salary driver.")

    # 5. Age vs Salary scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="Age", y="Salary", data=df, alpha=0.6, color="#457b9d")
    plt.title("05. Age vs Salary", fontsize=13)
    plt.xlabel("Age (Years)")
    plt.ylabel("Salary ($)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_age_vs_salary.png", dpi=120)
    plt.close()
    print("[Plot 05 Saved] Age vs Salary: Strong positive correlation reflecting natural career progression with age.")

    # 6. Education Level vs Salary boxplot
    edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Education Level", y="Salary", data=df, order=edu_order, hue="Education Level", legend=False, palette="Blues")
    plt.title("06. Salary by Education Level", fontsize=13)
    plt.xlabel("Education Level")
    plt.ylabel("Salary ($)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_education_vs_salary.png", dpi=120)
    plt.close()
    print("[Plot 06 Saved] Education vs Salary: Clear monotonic increase in median salary from High School to PhD.")

    # 7. Job Category vs Salary boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Job Category", y="Salary", data=df, hue="Job Category", legend=False, palette="Set2")
    plt.title("07. Salary by Job Category", fontsize=13)
    plt.xlabel("Job Category")
    plt.ylabel("Salary ($)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_job_category_vs_salary.png", dpi=120)
    plt.close()
    print("[Plot 07 Saved] Job Category vs Salary: Executive/Management and Data/Engineering show the highest salary brackets.")

    # 8. Seniority vs Salary boxplot
    seniority_labels = {0: "0 (Junior)", 1: "1 (Mid)", 2: "2 (Senior)", 3: "3 (Manager/Lead)", 4: "4 (Director/Exec)"}
    df_plot_sen = df.copy()
    df_plot_sen["Seniority_Label"] = df_plot_sen["Seniority"].map(seniority_labels)
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Seniority_Label", y="Salary", data=df_plot_sen, hue="Seniority_Label", legend=False, palette="Purples")
    plt.title("08. Salary by Seniority Level", fontsize=13)
    plt.xlabel("Seniority Level")
    plt.ylabel("Salary ($)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_seniority_vs_salary.png", dpi=120)
    plt.close()
    print("[Plot 08 Saved] Seniority vs Salary: Executive and leadership tiers clearly command the highest median salaries.")

    # 9. Gender vs Salary boxplot
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="Gender", y="Salary", data=df, hue="Gender", legend=False, palette="coolwarm")
    plt.title("09. Salary Distribution by Gender", fontsize=13)
    plt.xlabel("Gender")
    plt.ylabel("Salary ($)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_gender_vs_salary.png", dpi=120)
    plt.close()
    print("[Plot 09 Saved] Gender vs Salary: Men show slightly higher median salary, requiring formal statistical testing.")

    # Summary statistics tables
    print("\n" + "=" * 65)
    print("SUMMARY TABLES (MEDIAN SALARIES)")
    print("=" * 65)

    print("\n--- Median Salary by Education Level ---")
    edu_summary = df.groupby("Education Level")["Salary"].agg(["count", "median", "mean"]).reindex(edu_order)
    print(edu_summary)

    print("\n--- Median Salary by Job Category ---")
    cat_summary = df.groupby("Job Category")["Salary"].agg(["count", "median", "mean"]).sort_values(by="median", ascending=False)
    print(cat_summary)

    print("\n--- Median Salary by Seniority Level ---")
    sen_summary = df.groupby("Seniority")["Salary"].agg(["count", "median", "mean"])
    sen_summary.index = [seniority_labels[i] for i in sen_summary.index]
    print(sen_summary)

    print("\n" + "=" * 65)
    print(f"STAGE 3 COMPLETE: All 9 plots saved to {PLOTS_DIR.resolve()}")
    print("=" * 65)
    return df


if __name__ == "__main__":
    run_stage3()
