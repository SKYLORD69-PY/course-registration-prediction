import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from scripts.preprocess import preprocess_pipeline
from database.db_connection import get_engine
from project_config import (
    MODEL_BASE_FOLDER,
    RANDOM_STATE,
    TEST_SIZE
)


# -----------------------------
# Safe MAPE
# -----------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100


# -----------------------------
# Evaluate Model
# -----------------------------
def evaluate_model(version):

    print("Running preprocessing pipeline...\n")

    X, y = preprocess_pipeline()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    model_path = os.path.join(
        MODEL_BASE_FOLDER,
        version,
        "best_model.joblib"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found for version {version}")

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"\n📊 MODEL EVALUATION ({version})\n")

    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2  : {r2:.3f}")
    print(f"MAPE: {mape:.3f}%")

    return mae, rmse, r2, mape


# -----------------------------
# Log Model History
# -----------------------------
def log_model_history(mae, rmse, r2, mape, version):

    registry_path = os.path.join(
        MODEL_BASE_FOLDER,
        version,
        "model_registry.json"
    )

    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"❌ Registry not found for version {version}")

    with open(registry_path, "r") as f:
        registry = json.load(f)

    engine = get_engine()

    df = pd.DataFrame([{
        "model_version": version,
        "model_type": registry["best_model"],
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,

        # 🔥 IMPORTANT (dashboard consistency)
        "training_rows": registry.get("training_rows", None),

        "model_path": os.path.join(
            MODEL_BASE_FOLDER,
            version,
            "best_model.joblib"
        )
    }])

    df.to_sql(
        "model_history",
        engine,
        if_exists="append",
        index=False
    )

    print("✅ Model evaluation stored in database")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":

    version = "v_test"

    mae, rmse, r2, mape = evaluate_model(version)

    log_model_history(mae, rmse, r2, mape, version)