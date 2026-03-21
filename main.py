import os
import sys
import subprocess
from datetime import datetime
from database.db_connection import get_engine

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# ==============================
# CONFIG
# ==============================
MODEL_BASE_FOLDER = os.path.join(PROJECT_ROOT, "models")
ACTIVE_VERSION_FILE = os.path.join(MODEL_BASE_FOLDER, "active_version.txt")

def run_module(module_name):
    subprocess.run(
        [sys.executable, "-m", module_name],
        check=True,
        cwd=PROJECT_ROOT
    )
# ==============================
# STEP 1 — GENERATE DATA
# ==============================
def generate_data():
    print("\n📊 Generating dataset...\n")
    run_module("database.generate_dataset")


def load_data():
    print("\n💾 Loading dataset into DB (APPEND MODE)...\n")
    
    import subprocess
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.load_data",
        ],
        check=True,
        cwd=PROJECT_ROOT
    )


def train_model():
    print("\n🤖 Training models (v2)...\n")
    run_module("scripts.train")


def evaluate_model():
    print("\n📈 Evaluating model...\n")
    run_module("scripts.evaluate")


# ==============================
# STEP 5 — ACTIVATE VERSION
# ==============================
def activate_model(version="v2"):
    print(f"\n🚀 Activating {version}...\n")

    os.makedirs(MODEL_BASE_FOLDER, exist_ok=True)

    with open(ACTIVE_VERSION_FILE, "w") as f:
        f.write(version)

    print(f"✅ {version} is now ACTIVE")


# ==============================
# STEP 6 — RUN DASHBOARD
# ==============================
def run_dashboard():
    print("\n🌐 Launching Streamlit dashboard...\n")

    subprocess.run([
        "streamlit",
        "run",
        "dashboard/app.py"
    ])


# ==============================
# MAIN PIPELINE
# ==============================
if __name__ == "__main__":

    print("\n🔥 EDCAST AI — FULL PIPELINE START 🔥\n")

    try:
        generate_data()
        load_data()
        train_model()
        evaluate_model()
        activate_model("v2")

        print("\n✅ SYSTEM READY\n")

        run_dashboard()

    except subprocess.CalledProcessError as e:
        print("\n❌ PIPELINE FAILED:", e)