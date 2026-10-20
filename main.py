# -*- coding: utf-8 -*-
"""
Employee Salary Prediction -- Main Entry Point & Orchestrator
Data Exploration and Visualization Laboratory (DEVL) Mini-Project
=================================================================
Commands:
  python main.py                -> Executes all 5 pipeline stages end-to-end
  python main.py all            -> Executes all 5 pipeline stages end-to-end
  python main.py stage <1-5>    -> Executes an individual stage (e.g. python main.py stage 1)
  python main.py predict        -> Launches interactive salary prediction terminal CLI
  python main.py help           -> Displays CLI usage instructions
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stages.stage1_load_summary import run_stage1
from stages.stage2_preprocess import run_stage2
from stages.stage3_eda import run_stage3
from stages.stage4_stats import run_stage4
from stages.stage5_models import run_stage5, interactive_prediction


def run_all_stages():
    print("\n" + "#" * 70)
    print("# DEVL MINI-PROJECT: COMPLETE 5-STAGE END-TO-END PIPELINE")
    print("#" * 70 + "\n")

    print("\n>>> RUNNING STAGE 1 / 5: Data Ingestion & Summary...")
    run_stage1()

    print("\n>>> RUNNING STAGE 2 / 5: Cleaning, Feature Engineering & Encoding...")
    run_stage2()

    print("\n>>> RUNNING STAGE 3 / 5: Exploratory Data Analysis & Plots...")
    run_stage3()

    print("\n>>> RUNNING STAGE 4 / 5: Statistical Tests & Feature Importance...")
    run_stage4()

    print("\n>>> RUNNING STAGE 5 / 5: Regression Models & Evaluation...")
    run_stage5()

    print("\n" + "=" * 70)
    print("SUCCESS: All 5 stages executed cleanly. Plots & models saved to outputs/")
    print("=" * 70)
    print("\nTo predict salaries interactively, run:")
    print("  python main.py predict\n")


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
  python main.py predict        Launch interactive salary estimator CLI
  python main.py help           Show this help message
""")


def main():
    args = [arg.strip().lower() for arg in sys.argv[1:]]

    if not args or args[0] in ("all", "run", "pipeline"):
        run_all_stages()
    elif args[0] == "predict":
        interactive_prediction()
    elif args[0] in ("help", "--help", "-h"):
        print_help()
    elif args[0] == "stage":
        if len(args) < 2:
            print("Error: Missing stage number. Example: python main.py stage 1")
            sys.exit(1)
        stage_num = args[1]
        stage_map = {
            "1": run_stage1,
            "2": run_stage2,
            "3": run_stage3,
            "4": run_stage4,
            "5": run_stage5,
        }
        if stage_num in stage_map:
            stage_map[stage_num]()
        else:
            print(f"Error: Unknown stage '{stage_num}'. Choose between 1 and 5.")
            sys.exit(1)
    else:
        print(f"Unknown command: {' '.join(args)}")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
