"""
Central configuration file for the Hackathon 3 project.
All paths, model settings, and constants are defined here.
"""

import os

# ==============================
# Project Root
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================
# Data Paths
# ==============================

DATA_FOLDER = os.path.join(BASE_DIR, "data")

DATASET_FILE = os.path.join(
    DATA_FOLDER,
    "engineering_workshop_term_school_final_v2.csv"
)


# ==============================
# Database
# ==============================

DATABASE_FILE = os.path.join(BASE_DIR, "hackathon3.db")

DATABASE_URL = f"sqlite:///{DATABASE_FILE}"


# ==============================
# Model Paths
# ==============================

MODEL_FOLDER = os.path.join(BASE_DIR, "models", "saved_models")

MODEL_REGISTRY = os.path.join(BASE_DIR, "models", "model_registry.json")


# ==============================
# Artifacts
# ==============================

PLOTS_FOLDER = os.path.join(BASE_DIR, "artifacts", "plots")

REPORTS_FOLDER = os.path.join(BASE_DIR, "artifacts", "reports")


# ==============================
# Machine Learning Settings
# ==============================

TARGET_COLUMN = "enrollment_next_term"

FEATURE_COLUMNS = [
    "school",
    "term_label",
    "school_year1_population",
    "school_year2_population",
    "school_year3_population",
    "school_year4_population",
    "total_students_in_school",
    "avg_remaining_credits",
    "prev_term_enrollment",
    "prev2_term_enrollment",
    "recent_trend"
]


# ==============================
# Training Settings
# ==============================

TEST_SIZE = 0.2

RANDOM_STATE = 42