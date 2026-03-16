"""
Main pipeline controller for Hackathon 3 project.

This script orchestrates:
1. Database setup
2. Data loading
3. EDA generation
4. Model training (optional)
5. Model evaluation (optional)

Default run does NOT retrain the model or regenerate data.
"""

import os
import sys
import argparse

# ==============================
# Add project root to path
# ==============================

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

# ==============================
# Import modules
# ==============================

from scripts.load_data import create_tables, load_dataset
from scripts.eda import run_eda
from scripts.train import train_models
from scripts.evaluate import evaluate_model, log_model_history


# ==============================
# Pipeline Function
# ==============================

def run_pipeline(refresh_data=False, train=False, evaluate=False):

    print("\n==============================")
    print(" Hackathon 3 ML Pipeline ")
    print("==============================\n")

    # Step 1 — Create DB tables
    print("Creating database tables...")
    create_tables()

    # Step 2 — Load dataset
    print("\nLoading dataset...")
    load_dataset(overwrite=refresh_data)

    # Step 3 — Generate EDA artifacts
    print("\nRunning EDA...")
    run_eda()

    # Step 4 — Train models
    if train:
        print("\nTraining models...")
        train_models()
    else:
        print("\nSkipping model training")

    # Step 5 — Evaluate model
    if evaluate:
        print("\nEvaluating model...")
        mae, rmse, r2, mape = evaluate_model()

        log_model_history(mae, rmse, r2, mape)

    else:
        print("\nSkipping evaluation")

    print("\nPipeline finished successfully\n")
    print("To launch dashboard run:\n")
    print("streamlit run dashboard/app.py\n")


# ==============================
# CLI Entry
# ==============================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Recreate dataset and overwrite database"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train ML models"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate best model"
    )

    args = parser.parse_args()

    run_pipeline(
        refresh_data=args.refresh_data,
        train=args.train,
        evaluate=args.evaluate
    )