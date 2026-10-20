# -*- coding: utf-8 -*-
"""
Employee Salary Prediction -- Main Entry Point & Orchestrator
Data Exploration and Visualization Laboratory (DEVL) Mini-Project
=================================================================
Usage:
  python main.py                -> Runs all 5 pipeline stages end-to-end
  python main.py all            -> Runs all 5 pipeline stages end-to-end
  python main.py stage <1-5>    -> Runs an individual stage (e.g. python main.py stage 1)
  python main.py predict        -> Launches interactive salary prediction terminal CLI
  python main.py test           -> Executes automated 46-point smoke test suite
  python main.py help           -> Shows CLI usage instructions
"""

import os
os.environ["MPLBACKEND"] = "Agg"  # Force headless non-blocking plotting backend

import sys
from pathlib import Path

# Enable unbuffered / line-buffered stdout for real-time terminal output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_stage(stage_num: str):
    """Execute a single pipeline stage with lazy-loaded imports."""
    stage_num = str(stage_num).strip()
    if stage_num == "1":
        print("\n>>> Loading Stage 1 (Data Ingestion & Summary)...", flush=True)
        from stages.stage1_load_summary import run_stage1
        run_stage1()
    elif stage_num == "2":
        print("\n>>> Loading Stage 2 (Cleaning, Feature Engineering & Encoding)...", flush=True)
        from stages.stage2_preprocess import run_stage2
        run_stage2()
    elif stage_num == "3":
        print("\n>>> Loading Stage 3 (Exploratory Data Analysis & Plots)...", flush=True)
        from stages.stage3_eda import run_stage3
        run_stage3()
    elif stage_num == "4":
        print("\n>>> Loading Stage 4 (Statistical Tests & Feature Importance)...", flush=True)
        from stages.stage4_stats import run_stage4
        run_stage4()
    elif stage_num == "5":
        print("\n>>> Loading Stage 5 (Model Training, Evaluation & Benchmarking)...", flush=True)
        from stages.stage5_models import run_stage5
        run_stage5()
    else:
        print(f"Error: Unknown stage '{stage_num}'. Choose between 1 and 5.", flush=True)
        sys.exit(1)


def run_all_stages():
    """Execute all 5 stages end-to-end with real-time console feedback."""
    print("\n" + "=" * 70, flush=True)
    print("  EMPLOYEE SALARY PREDICTION -- 5-STAGE DEVL PIPELINE", flush=True)
    print("=" * 70, flush=True)

    print("\n[1/5] Executing Stage 1: Data Ingestion & Summary...", flush=True)
    from stages.stage1_load_summary import run_stage1
    run_stage1()

    print("\n[2/5] Executing Stage 2: Data Cleaning, Feature Engineering & Encoding...", flush=True)
    from stages.stage2_preprocess import run_stage2
    run_stage2()

    print("\n[3/5] Executing Stage 3: Exploratory Data Analysis & Visualizations...", flush=True)
    from stages.stage3_eda import run_stage3
    run_stage3()

    print("\n[4/5] Executing Stage 4: Statistical Hypothesis Testing & Feature Importance...", flush=True)
    from stages.stage4_stats import run_stage4
    run_stage4()

    print("\n[5/5] Executing Stage 5: Regressors, Evaluation & Benchmarking...", flush=True)
    from stages.stage5_models import run_stage5
    run_stage5()

    print("\n" + "=" * 70, flush=True)
    print("  PIPELINE COMPLETE: All 5 stages executed successfully!", flush=True)
    print("  Datasets saved to : data/processed/", flush=True)
    print("  13 Plots saved to : outputs/plots/", flush=True)
    print("  Model saved to    : outputs/models/salary_rf_model.joblib", flush=True)
    print("=" * 70, flush=True)
    print("\nTo predict salaries interactively, run:\n  python main.py predict\n", flush=True)


def print_help():
    print("""
Employee Salary Prediction - CLI Usage
======================================
Usage:
  python main.py                Run all 5 pipeline stages sequentially
  python main.py all            Run all 5 pipeline stages sequentially
  python main.py stage <1-5>    Run a single pipeline stage:
                                  1 = Load & Dataset Summary
                                  2 = Preprocessing & Feature Engineering
                                  3 = EDA & Visualization Charts
                                  4 = Statistical Inference (t-test, ANOVA)
                                  5 = Model Training, Evaluation & Benchmarking
  python main.py predict        Launch interactive terminal salary predictor
  python main.py test           Run automated 46-point smoke test suite
  python main.py help           Show this help message
""", flush=True)


def main():
    args = [arg.strip().lower() for arg in sys.argv[1:]]

    if not args or args[0] in ("all", "run", "pipeline"):
        run_all_stages()
    elif args[0] == "predict":
        from stages.stage5_models import interactive_prediction
        interactive_prediction()
    elif args[0] in ("test", "smoke", "test_smoke"):
        import test_smoke
    elif args[0] in ("help", "--help", "-h"):
        print_help()
    elif args[0] == "stage":
        if len(args) < 2:
            print("Error: Missing stage number. Example: python main.py stage 1", flush=True)
            sys.exit(1)
        run_stage(args[1])
    else:
        print(f"Unknown command: {' '.join(args)}", flush=True)
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
