import os
import sys

# -----------------------------
# Add project root to path
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

# -----------------------------
# Import project modules
# -----------------------------
from scripts.load_data import create_tables, load_dataset
from scripts.eda import run_eda
from scripts.train import train_models
from scripts.evaluate import evaluate_model, log_model_history


# -----------------------------
# Main Pipeline
# -----------------------------
def run_pipeline():

    print("\n===================================")
    print(" Hackathon 3 ML Pipeline Starting ")
    print("===================================\n")

    # Step 1: Create database tables
    print("Step 1: Creating database tables...")
    create_tables()

    # Step 2: Load dataset
    print("\nStep 2: Loading dataset into SQLite...")
    load_dataset()

    # Step 3: Run EDA
    print("\nStep 3: Running EDA analysis...")
    run_eda()

    # Step 4: Train ML models
    print("\nStep 4: Training models...")
    train_models()

    # Step 5: Evaluate best model
    print("\nStep 5: Evaluating best model...")
    mae, rmse, r2, mape = evaluate_model()

    log_model_history(mae, rmse, r2, mape)

    print("\n===================================")
    print(" Pipeline Completed Successfully ")
    print("===================================\n")

    print("Next step:")
    print("Run dashboard with:\n")
    print("streamlit run dashboard/app.py\n")


# -----------------------------
# Run pipeline
# -----------------------------
if __name__ == "__main__":

    run_pipeline()