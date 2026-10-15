"""
Employee Salary Prediction -- Main Entry Point
===============================================
Run this file to execute the complete project workflow:
  python main.py          -> Full pipeline (inspection -> training -> evaluation)
  python main.py predict  -> Interactive salary prediction using the trained model

Data Exploration and Visualization Laboratory (DEVL) Mini-Project
"""

import sys
from pathlib import Path

# Ensure project root is on the path so 'src' can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.salary_prediction import run_full_pipeline, interactive_prediction


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "train"

    if mode == "predict":
        # Launch interactive prediction using saved model
        interactive_prediction()
    else:
        # Run full pipeline: inspect -> clean -> EDA -> stats -> train -> evaluate
        pipeline, metrics = run_full_pipeline()
        if pipeline is not None:
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETE -- All steps executed successfully.")
            print("=" * 60)
            print("\nTo predict salaries interactively, run:")
            print("  python main.py predict")


if __name__ == "__main__":
    main()
