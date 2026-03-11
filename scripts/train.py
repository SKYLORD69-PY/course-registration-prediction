import os
import sys
import json
import joblib
import numpy as np

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
from project_config import MODEL_FOLDER, RANDOM_STATE, TEST_SIZE


# -----------------------------
# Safe MAPE
# -----------------------------
def mean_absolute_percentage_error(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100


# -----------------------------
# Train Models
# -----------------------------
def train_models():

    print("Running preprocessing pipeline...\n")

    X, y = preprocess_pipeline()

    categorical_cols = ["school", "term_label"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    results = {}

    print("MODEL TRAINING RESULTS\n")

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print(f"Model: {name}")
        print(f"MAE : {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2  : {r2:.3f}")
        print(f"MAPE: {mape:.3f}%")
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
    best_pipeline = results[best_model_name]["pipeline"]

    print("\nBest Model Selected:", best_model_name)

    # -----------------------------
    # Save Model
    # -----------------------------
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    model_path = os.path.join(MODEL_FOLDER, "best_model.joblib")

    joblib.dump(best_pipeline, model_path)

    print("Model saved to:", model_path)

    # -----------------------------
    # Save Model Registry JSON
    # -----------------------------
    best_metrics = results[best_model_name]

    registry = {
        "model_name": "Workshop Enrollment Predictor",
        "version": "v1",
        "description": "Predicts the number of students enrolling in Engineering Workshop in the next academic term",
        "algorithms": [
            "Linear Regression",
            "Ridge Regression",
            "Random Forest Regressor",
            "Gradient Boosting Regressor"
        ],
        "best_model": best_model_name,
        "metrics": {
            "mae": float(best_metrics["mae"]),
            "rmse": float(best_metrics["rmse"]),
            "r2": float(best_metrics["r2"]),
            "mape": float(best_metrics["mape"])
        },
        "note": "Binary model file is generated locally via scripts/train.py"
    }

    registry_path = os.path.join(MODEL_FOLDER, "model_registry.json")

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)

    print("Model registry saved to:", registry_path)


# -----------------------------
# Run Training
# -----------------------------
if __name__ == "__main__":

    train_models()