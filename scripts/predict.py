import os
import sys
import joblib
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from project_config import MODEL_BASE_FOLDER, ACTIVE_VERSION_FILE, MODEL_VERSION

def get_active_version():
    if os.path.exists(ACTIVE_VERSION_FILE):
        with open(ACTIVE_VERSION_FILE, "r") as f:
            v = f.read().strip()
            if v:
                return v
    return MODEL_VERSION

def load_model(version=None):
    version = version or get_active_version()
    model_path = os.path.join(MODEL_BASE_FOLDER, version, "best_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for version {version}. Train first.")
    return joblib.load(model_path)

def predict(input_dict, version=None):
    model = load_model(version=version)
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    return round(float(prediction), 2)