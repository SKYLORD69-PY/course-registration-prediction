import os
import sys
import joblib
import pandas as pd

# -----------------------------
# Add project root
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from project_config import MODEL_FOLDER


# -----------------------------
# Load Model
# -----------------------------
def load_model():

    model_path = os.path.join(MODEL_FOLDER, "best_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run train.py first.")

    model = joblib.load(model_path)

    return model


# -----------------------------
# Make Prediction
# -----------------------------
def predict_enrollment(input_data):

    model = load_model()

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]

    return round(prediction, 2)


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":

    example_input = {

        "school": "VSST",
        "term_label": "independence",

        "school_year1_population": 60,
        "school_year2_population": 58,
        "school_year3_population": 54,
        "school_year4_population": 48,

        "total_students_in_school": 220,

        "avg_remaining_credits": 27,

        "prev_term_enrollment": 38,
        "prev2_term_enrollment": 42,
        "recent_trend": -4
    }

    prediction = predict_enrollment(example_input)

    print("\nPredicted Enrollment for Next Term:", prediction)