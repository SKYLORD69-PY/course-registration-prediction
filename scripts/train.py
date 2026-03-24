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
def train_models(version):

    print(f"\n🚀 Training model version: {version}\n")

    # -----------------------------
    # Load Data
    # -----------------------------
    X, y = preprocess_pipeline()

    # -----------------------------
    # Feature Split
    # -----------------------------
    categorical_cols = ["school", "term_label"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ])

    # -----------------------------
    # Models
    # -----------------------------
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    results = {}

    print("MODEL TRAINING RESULTS\n")

    # -----------------------------
    # Train All Models
    # -----------------------------
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

        print(f"{name} → R2={r2:.3f}")

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
    best_pipeline = results[best_model_name]["pipeline"]

    print(f"\n🏆 Best Model: {best_model_name}")

    # -----------------------------
    # Save Model
    # -----------------------------
    model_dir = os.path.join(MODEL_BASE_FOLDER, version)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "best_model.joblib")
    joblib.dump(best_pipeline, model_path)

    # -----------------------------
    # Create Registry
    # -----------------------------
    registry = {
        "model_name": "Workshop Enrollment Predictor",
        "version": version,

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

        # 🔥 VERY IMPORTANT (for dashboard correctness)
        "training_rows": int(len(X)),

        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save registry JSON
    registry_path = os.path.join(model_dir, "model_registry.json")
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)

    # -----------------------------
    # Update Master Registry
    # -----------------------------
    if os.path.exists(MASTER_REGISTRY):
        with open(MASTER_REGISTRY, "r") as f:
            master = json.load(f)
    else:
        master = []

    # remove duplicate version if exists
    master = [m for m in master if m["version"] != version]

    master.append({
        "version": version,
        "best_model": best_model_name,
        "metrics": registry["metrics"],
        "training_rows": registry["training_rows"],
        "trained_at": registry["trained_at"],
        "path": model_dir
    })

    with open(MASTER_REGISTRY, "w") as f:
        json.dump(master, f, indent=4)

    print("✅ Training complete\n")

    return registry


# -----------------------------
# Run (for testing only)
# -----------------------------
if __name__ == "__main__":
    train_models(version="v_test")