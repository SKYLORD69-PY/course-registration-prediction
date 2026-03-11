import os
import sys
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from database.db_connection import get_engine
from project_config import FEATURE_COLUMNS, TARGET_COLUMN


# -----------------------------
# Load Data From SQLite
# -----------------------------
def load_data():

    engine = get_engine()

    query = "SELECT * FROM terms_enrollment"

    df = pd.read_sql(query, engine)

    print(f"Loaded {len(df)} rows from database")

    return df


# -----------------------------
# Create Prediction Target
# -----------------------------
def create_target(df):

    df = df.sort_values(["school", "term_start_date"])

    df["enrollment_next_term"] = (
        df.groupby("school")["enrollment_this_term"].shift(-1)
    )

    df = df.dropna(subset=["enrollment_next_term"])

    df["enrollment_next_term"] = df["enrollment_next_term"].astype(int)

    return df


# -----------------------------
# Remove Leakage Columns
# -----------------------------
def remove_leakage_columns(df):

    leakage_cols = [
        "enrolled_year1",
        "enrolled_year2",
        "enrolled_year3",
        "enrolled_year4",
        "num_compulsory",
        "num_ge",
        "num_elective",
        "num_other"
    ]

    df = df.drop(columns=leakage_cols)

    return df


# -----------------------------
# Drop Unnecessary Columns
# -----------------------------
def drop_unused_columns(df):

    drop_cols = [
        "year",
        "term_start_date"
    ]

    df = df.drop(columns=drop_cols)

    return df


# -----------------------------
# Prepare Features
# -----------------------------
def prepare_features(df):

    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMN]

    return X, y


# -----------------------------
# Main Preprocess Pipeline
# -----------------------------
def preprocess_pipeline():

    df = load_data()

    df = create_target(df)

    df = remove_leakage_columns(df)

    df = drop_unused_columns(df)

    X, y = prepare_features(df)

    print("\nPreprocessing completed")

    print("Feature shape:", X.shape)

    print("Target shape:", y.shape)

    return X, y


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":

    X, y = preprocess_pipeline()

    print("\nSample features:")
    print(X.head())

    print("\nSample target:")
    print(y.head())