# dashboard/app.py
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

# paths to artifact images (these are the exact visuals saved by your EDA)
PLOTS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "plots")
PLOT_FILES = {
    "enrollment_by_school": os.path.join(PLOTS_DIR, "enrollment_by_school.png"),
    "enrollment_by_term": os.path.join(PLOTS_DIR, "enrollment_by_term.png"),
    "enrollment_trend": os.path.join(PLOTS_DIR, "enrollment_trend.png"),
    "correlation_heatmap": os.path.join(PLOTS_DIR, "correlation_heatmap.png"),
}

REGISTRY_PATH = os.path.join(MODEL_FOLDER, "model_registry.json")
FEEDBACK_PATH = os.path.join(PROJECT_ROOT, "artifacts", "feedback.csv")

# -----------------------
# Streamlit page config
# -----------------------
st.set_page_config(
    page_title="Workshop Enrollment Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎓 Engineering Workshop — Enrollment Predictor")
st.markdown(
    "Compact dashboard for students & staff — insights, model info, and a small prediction tool."
)

# -----------------------
# Load data (for quick summaries)
# -----------------------
engine = get_engine()
df = pd.read_sql("SELECT * FROM terms_enrollment", engine)

# -----------------------
# Top summary (for regular users)
# -----------------------
st.header("At a glance")
col_a, col_b, col_c = st.columns([1, 1, 1.2])
with col_a:
    avg_enrollment = round(df["enrollment_this_term"].mean(), 1)
    st.metric("Avg enrollment (per term)", f"{avg_enrollment} students")
with col_b:
    max_school = df.groupby("school")["enrollment_this_term"].mean().idxmax()
    max_school_val = round(df.groupby("school")["enrollment_this_term"].mean().max(), 1)
    st.metric("Most popular school (avg)", f"{max_school} — {max_school_val}")
with col_c:
    avg_remaining = round(df["avg_remaining_credits"].mean(), 1)
    st.metric("Avg remaining credits", f"{avg_remaining}")

st.markdown(
    "Short note: values are synthetic for the project. Use the **Predict** tab to estimate how many students will register next term."
)

st.markdown("---")

# -----------------------
# EDA: 2x2 grid of artifact images (not huge)
# -----------------------
st.header("Insights")
st.markdown("Key visualizations — these are the same figures saved in `artifacts/plots/`.")

# create two rows with two images each
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

IMAGE_WIDTH = 420  # reasonable size: not huge, not tiny

def show_image(col, path, caption):
    if os.path.exists(path):
        col.image(path, caption=caption, width=IMAGE_WIDTH)
    else:
        col.warning(f"Missing image: {os.path.basename(path)}")

show_image(row1_col1, PLOT_FILES["enrollment_by_school"], "Average enrollment by school")
show_image(row1_col2, PLOT_FILES["enrollment_by_term"], "Average enrollment by term")
show_image(row2_col1, PLOT_FILES["enrollment_trend"], "Enrollment trend over time")
show_image(row2_col2, PLOT_FILES["correlation_heatmap"], "Feature correlation heatmap")

st.markdown("---")

# -----------------------
# Model info (compact, collapsible)
# -----------------------
st.header("Model info (for curious users)")
if os.path.exists(REGISTRY_PATH):
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)
else:
    registry = None

with st.expander("Show model details"):
    if registry:
        st.subheader(registry.get("model_name", "Model"))
        st.write(registry.get("description", "No description available."))
        st.write("Version:", registry.get("version", "n/a"))
        st.write("Algorithms tried:", ", ".join(registry.get("algorithms", [])))
        best_model = registry.get("best_model", "n/a")
        st.write("Best model:", best_model)
        metrics = registry.get("metrics", {})
        # Accuracy score (use R² as an 'accuracy' indicator for users)
        r2 = metrics.get("r2", None)
        if r2 is not None:
            acc_pct = round(r2 * 100, 2)
            st.metric("Accuracy (R²)", f"{acc_pct}%")
        # concise metrics
        st.write(
            f"MAE: {metrics.get('mae','-')}  ·  RMSE: {metrics.get('rmse','-')}  ·  SMAPE/Note: MAPE can be unstable for small counts"
        )
    else:
        st.info("Model registry not found — run training first (scripts/train.py).")

st.markdown("---")

# -----------------------
# Predict section (compact UI)
# -----------------------
st.header("Quick prediction")
st.markdown("Enter a few values and press **Predict**. Keep inputs short and simple.")

# left: school + term, middle: populations, right: recent stats
c1, c2, c3 = st.columns([1, 1.1, 1])

with c1:
    school = st.selectbox("School", ["VSST", "TSM", "JAGSoM", "VSOD", "VSOL"])
    term_label = st.selectbox("Term", ["independence", "festivals", "republic", "colors"])

with c2:
    # compact numeric inputs for populations
    st.caption("Students per year (compact inputs)")
    y1 = st.number_input("Year1", min_value=0, max_value=500, value=60, step=1, key="y1")
    y2 = st.number_input("Year2", min_value=0, max_value=500, value=55, step=1, key="y2")
    y3 = st.number_input("Year3", min_value=0, max_value=500, value=50, step=1, key="y3")
    y4 = st.number_input("Year4", min_value=0, max_value=500, value=45, step=1, key="y4")

with c3:
    # compact recent stats
    st.caption("Recent enrollments")
    prev1 = st.number_input("Previous term", min_value=0, value=35, step=1, key="prev1")
    prev2 = st.number_input("Two terms ago", min_value=0, value=40, step=1, key="prev2")
    credits = st.number_input("Avg remaining credits", min_value=0, max_value=40, value=27, step=1, key="credits")

# compute derived values
total_students = int(y1 + y2 + y3 + y4)
recent_trend = int(prev1 - prev2)

# compact predict button
if st.button("Predict enrollment (next term)"):
    input_data = {
        "school": school,
        "term_label": term_label,
        "school_year1_population": int(y1),
        "school_year2_population": int(y2),
        "school_year3_population": int(y3),
        "school_year4_population": int(y4),
        "total_students_in_school": total_students,
        "avg_remaining_credits": float(credits),
        "prev_term_enrollment": int(prev1),
        "prev2_term_enrollment": int(prev2),
        "recent_trend": int(recent_trend),
    }

    try:
        pred = predict_enrollment(input_data)
        st.success(f"Predicted students next term: **{pred}**")
        st.info("Note: this prediction uses the current trained model and synthetic historic trends.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")

# -----------------------
# Feedback (compact)
# -----------------------
st.header("Feedback")
st.markdown("If something looks off or you'd like to report a problem, leave a short note.")

with st.form("feedback_form", clear_on_submit=True):
    fb_name = st.text_input("Your name (optional)")
    fb_text = st.text_area("Feedback (short)", max_chars=500, placeholder="UI glitch, chart suggestion, or prediction oddity...")
    submitted = st.form_submit_button("Send feedback")
    if submitted:
        os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
        # append to CSV
        row = {"name": fb_name or "anonymous", "feedback": fb_text}
        if os.path.exists(FEEDBACK_PATH):
            fb_df = pd.read_csv(FEEDBACK_PATH)
            fb_df = fb_df.append(row, ignore_index=True)
        else:
            fb_df = pd.DataFrame([row])
        fb_df.to_csv(FEEDBACK_PATH, index=False)
        st.success("Thanks — feedback recorded.")

# -----------------------
# Footer - small tips
# -----------------------
st.markdown("---")
st.caption(
    "Tip: For classroom managers, try different 'prev term' values to see how the forecast reacts. "
    "If the dashboard or model seems off, re-run the pipeline: `python main.py`."
)