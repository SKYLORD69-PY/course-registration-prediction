import os
import sys
import json
import joblib
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Add project root
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from scripts.preprocess import preprocess_pipeline
from project_config import (
    MODEL_BASE_FOLDER,
    MASTER_REGISTRY,
    ACTIVE_VERSION_FILE,
    RANDOM_STATE,
    TEST_SIZE
)


# -----------------------------
# Safe MAPE
# -----------------------------
def safe_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100


# -----------------------------
# Train Models
# -----------------------------
def train_models(model_version="v2", activate=True):

    print(f"\n🚀 Training model version: {model_version}\n")

    X, y = preprocess_pipeline()

    categorical_cols = ["school", "term_label"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ])

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = {}

    print("MODEL TRAINING RESULTS\n")

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = safe_mape(y_test, y_pred)

        print(f"Model: {name}")
        print(f"MAE : {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2  : {r2:.3f}")
        print(f"MAPE: {mape:.2f}%")
        print("-" * 30)

        results[name] = {
            "pipeline": pipeline,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        }

    # -----------------------------
    # Select Best Model
    # -----------------------------
    best_model_name = max(results, key=lambda x: results[x]["r2"])
    best_model = results[best_model_name]["pipeline"]

    print("\nBest Model Selected:", best_model_name)

    # -----------------------------
    # Save Model (versioned)
    # -----------------------------
    model_dir = os.path.join(MODEL_BASE_FOLDER, model_version)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)

    print("Model saved to:", model_path)

    # -----------------------------
    # Save Registry
    # -----------------------------
    registry = {
        "model_name": "Workshop Enrollment Predictor",
        "version": model_version,
        "best_model": best_model_name,
        "metrics": {
            "mae": results[best_model_name]["mae"],
            "rmse": results[best_model_name]["rmse"],
            "r2": results[best_model_name]["r2"],
            "mape": results[best_model_name]["mape"]
        },
        "all_models": {
            name: {
                "mae": results[name]["mae"],
                "rmse": results[name]["rmse"],
                "r2": results[name]["r2"],
                "mape": results[name]["mape"]
            }
            for name in results
        },
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    registry_path = os.path.join(model_dir, "model_registry.json")

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)

    print("Model registry saved to:", registry_path)

    # -----------------------------
    # Update Master Registry
    # -----------------------------
    if os.path.exists(MASTER_REGISTRY):
        with open(MASTER_REGISTRY, "r") as f:
            master = json.load(f)
    else:
        master = []

    master.append({
        "version": model_version,
        "best_model": best_model_name,
        "metrics": registry["metrics"],
        "trained_at": registry["trained_at"],
        "path": model_dir
    })

    with open(MASTER_REGISTRY, "w") as f:
        json.dump(master, f, indent=4)

    print("Master registry updated")

    # -----------------------------
    # Set Active Version
    # -----------------------------
    if activate:
        with open(ACTIVE_VERSION_FILE, "w") as f:
            f.write(model_version)

        print(f"Active model set to {model_version}")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    train_models(model_version="v2", activate=True)