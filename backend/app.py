import os
import sys
import json
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from database.db_connection import get_engine
from scripts.predict import predict_enrollment
from project_config import MODEL_FOLDER


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Workshop Enrollment Predictor",
    layout="wide"
)

st.title("🎓 Engineering Workshop Enrollment Prediction Dashboard")


# -----------------------------
# Load Dataset
# -----------------------------
engine = get_engine()

df = pd.read_sql("SELECT * FROM terms_enrollment", engine)


# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.selectbox(
    "Navigation",
    ["Dataset Overview", "EDA Insights", "Model Performance", "Predict Enrollment"]
)


# -----------------------------
# DATASET OVERVIEW
# -----------------------------
if page == "Dataset Overview":

    st.header("Dataset Overview")

    st.write("Total rows:", len(df))

    st.dataframe(df.head(20))


# -----------------------------
# EDA INSIGHTS
# -----------------------------
elif page == "EDA Insights":

    st.header("Exploratory Data Analysis")

    st.subheader("Enrollment by School")

    school_data = df.groupby("school")["enrollment_this_term"].mean()

    st.bar_chart(school_data)

    st.subheader("Enrollment by Term")

    term_data = df.groupby("term_label")["enrollment_this_term"].mean()

    st.bar_chart(term_data)

    st.subheader("Enrollment Trend")

    trend_data = df.groupby("term_start_date")["enrollment_this_term"].sum()

    st.line_chart(trend_data)


# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
elif page == "Model Performance":

    st.header("Model Performance")

    registry_path = os.path.join(MODEL_FOLDER, "model_registry.json")

    with open(registry_path, "r") as f:
        registry = json.load(f)

    st.subheader("Model Information")

    st.write("Model Name:", registry["model_name"])
    st.write("Version:", registry["version"])
    st.write("Best Algorithm:", registry["best_model"])

    st.subheader("Metrics")

    metrics = registry["metrics"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE", round(metrics["mae"], 2))
    col2.metric("RMSE", round(metrics["rmse"], 2))
    col3.metric("R²", round(metrics["r2"], 3))
    col4.metric("MAPE", f'{round(metrics["mape"],2)} %')


# -----------------------------
# PREDICTION
# -----------------------------
elif page == "Predict Enrollment":

    st.header("Predict Next Term Enrollment")

    school = st.selectbox(
        "School",
        ["VSST", "TSM", "JAGSoM", "VSOD", "VSOL"]
    )

    term = st.selectbox(
        "Term",
        ["independence", "festivals", "republic", "colors"]
    )

    col1, col2 = st.columns(2)

    with col1:
        y1 = st.number_input("Year 1 Students", value=60)
        y2 = st.number_input("Year 2 Students", value=55)
        y3 = st.number_input("Year 3 Students", value=50)
        y4 = st.number_input("Year 4 Students", value=45)

    with col2:
        credits = st.slider("Average Remaining Credits", 0, 40, 20)
        prev1 = st.number_input("Previous Term Enrollment", value=35)
        prev2 = st.number_input("2 Terms Ago Enrollment", value=40)

    total_students = y1 + y2 + y3 + y4

    trend = prev1 - prev2

    if st.button("Predict Enrollment"):

        input_data = {
            "school": school,
            "term_label": term,
            "school_year1_population": y1,
            "school_year2_population": y2,
            "school_year3_population": y3,
            "school_year4_population": y4,
            "total_students_in_school": total_students,
            "avg_remaining_credits": credits,
            "prev_term_enrollment": prev1,
            "prev2_term_enrollment": prev2,
            "recent_trend": trend
        }

        prediction = predict_enrollment(input_data)

        st.success(f"Predicted Enrollment: **{prediction} students**")