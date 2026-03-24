# main.py

import os
import sys
import subprocess

# ==============================
# Add project root
# ==============================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

# ==============================
# Imports
# ==============================
from scripts.load_data import create_tables, load_dataset
from scripts.train import train_models
from scripts.evaluate import evaluate_model, log_model_history
from scripts.eda import run_eda

from project_config import MODEL_BASE_FOLDER, ACTIVE_VERSION_FILE


# ==============================
# Version Generator (FINAL)
# ==============================
def generate_new_version():
    """
    Simple and correct versioning:
    - First run → v1
    - Next → v2, v3...
    - Ignores active_version.txt
    """

    if not os.path.exists(MODEL_BASE_FOLDER):
        return "v1"

    versions = []

    for folder in os.listdir(MODEL_BASE_FOLDER):
        if folder.startswith("v") and folder[1:].isdigit():
            versions.append(int(folder[1:]))

    if not versions:
        return "v1"

    return f"v{max(versions) + 1}"


# ==============================
# Set Active Version
# ==============================
def set_active_version(version):
    os.makedirs(MODEL_BASE_FOLDER, exist_ok=True)

    with open(ACTIVE_VERSION_FILE, "w") as f:
        f.write(version)

    print(f"🔥 ACTIVE MODEL SET TO: {version}")


# ==============================
# Launch Dashboard
# ==============================
def launch_dashboard():
    print("\n🌐 Launching Dashboard...\n")

    dashboard_path = os.path.join(PROJECT_ROOT, "dashboard", "app.py")

    try:
        subprocess.Popen([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            dashboard_path
        ])
    except Exception as e:
        print("❌ Failed to launch dashboard:", e)


# ==============================
# MAIN PIPELINE
# ==============================
def run_pipeline():

    print("\n" + "=" * 60)
    print("🚀 FULL ML PIPELINE STARTED")
    print("=" * 60 + "\n")

    # STEP 1: DB SETUP
    print("📦 STEP 1: Setting up database...")
    create_tables()

    # STEP 2: LOAD DATA
    print("\n📊 STEP 2: Loading new dataset...")
    load_dataset(
        append=True,
        dataset_version=None,
        seed=None,
        note="Auto run via main.py"
    )

    # STEP 3: VERSION
    print("\n🧠 STEP 3: Creating new model version...")
    version = generate_new_version()

    model_dir = os.path.join(MODEL_BASE_FOLDER, version)
    os.makedirs(model_dir, exist_ok=True)

    print(f"📁 New version created: {version}")

    # STEP 4: TRAIN
    print("\n⚙️ STEP 4: Training models...")
    train_models(version=version)

    # STEP 5: EVALUATE
    print("\n📈 STEP 5: Evaluating model...")
    mae, rmse, r2, mape = evaluate_model(version=version)

    # STEP 6: LOG
    print("\n🧾 STEP 6: Logging model history...")
    log_model_history(mae, rmse, r2, mape, version)

    # STEP 7: SET ACTIVE
    print("\n🔥 STEP 7: Updating active model...")
    set_active_version(version)

    # STEP 8: EDA
    print("\n📊 STEP 8: Running EDA...")
    try:
        run_eda()
    except Exception as e:
        print("⚠️ EDA skipped:", e)

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")

    # STEP 9: DASHBOARD
    launch_dashboard()


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    run_pipeline()