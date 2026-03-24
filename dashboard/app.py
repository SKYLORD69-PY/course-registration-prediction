import os
import sys
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# =========================================================
# Project root + config
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from project_config import (
    PLOTS_FOLDER,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    MODEL_BASE_FOLDER,
    ACTIVE_VERSION_FILE,
)
from database.db_connection import get_engine

# =========================================================
# Streamlit setup
# =========================================================
st.set_page_config(
    page_title="EduCast AI — Campus Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# Constants
# =========================================================
DISPLAY_TERM_ORDER = ["independence", "festivals", "republic", "colors"]
TERM_LABELS = {
    "independence": "Independence",
    "festivals": "Festivals",
    "republic": "Republic",
    "colors": "Colors",
}
COLORWAY = ["#F5C842", "#00E5C3", "#FF6B6B", "#A855F7", "#38BDF8", "#FB923C", "#4ADE80"]

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(20,24,55,0.5)",
    font=dict(family="DM Sans", color="#8B90B8", size=11),
    title_font=dict(family="Syne", color="#F0F2FF", size=14),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.08)",
        tickcolor="#5A5F80",
        showgrid=True,
        tickfont=dict(color="#8B90B8"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.08)",
        tickcolor="#5A5F80",
        showgrid=True,
        tickfont=dict(color="#8B90B8"),
    ),
    colorway=COLORWAY,
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        bgcolor="rgba(20,24,55,0.8)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        font=dict(color="#8B90B8"),
    ),
)

# =========================================================
# CSS
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --navy:   #0B0F2E;
    --navy2:  #141837;
    --gold:   #F5C842;
    --coral:  #FF6B6B;
    --mint:   #00E5C3;
    --violet: #A855F7;
    --sky:    #38BDF8;
    --white:  #F0F2FF;
    --muted:  #8B90B8;
    --border: rgba(255,255,255,0.08);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--navy) !important;
    color: var(--white) !important;
}
.block-container { padding: 0.5rem 2rem 2rem !important; }

.hero {
    background: linear-gradient(135deg, #0B0F2E 0%, #141837 40%, #1a1040 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 44px 48px 38px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(245,200,66,0.08) 0%, transparent 65%);
    top: -200px; right: -100px;
    border-radius: 50%;
    pointer-events: none;
}
.hero::after {
    content: "";
    position: absolute;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,229,195,0.07) 0%, transparent 65%);
    bottom: -100px; left: 100px;
    border-radius: 50%;
    pointer-events: none;
}
.logo {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.25em;
    color: var(--gold);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.logo::before {
    content: "";
    display: inline-block;
    width: 28px; height: 2px;
    background: var(--gold);
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 12px;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.hero h1 span { color: var(--gold); }
.hero p {
    color: var(--muted);
    font-size: 1rem;
    margin: 0 0 30px;
    font-weight: 400;
    max-width: 760px;
}
.hero-stats {
    display: flex;
    gap: 30px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
    flex-wrap: wrap;
}
.hstat .hv {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--gold);
    line-height: 1;
}
.hstat .hl {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-top: 5px;
}

.kcard {
    flex: 1;
    background: var(--navy2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 16px;
    position: relative;
    overflow: hidden;
    min-width: 170px;
}
.kcard::before {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, var(--gold));
}
.kcard .ki { font-size: 1.3rem; margin-bottom: 10px; }
.kcard .kv {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--accent, var(--gold));
    line-height: 1;
}
.kcard .kl {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-top: 6px;
}
.kcard .ks { font-size: 0.78rem; color: #5A5F80; margin-top: 3px; }

.panel {
    background: var(--navy2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 24px 26px;
    margin-bottom: 18px;
}
.panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 4px;
}
.panel-sub {
    font-size: 0.82rem;
    color: var(--muted);
    margin: 0 0 18px;
}

.finding {
    background: rgba(245,200,66,0.07);
    border-left: 3px solid var(--gold);
    border-radius: 0 10px 10px 0;
    padding: 11px 16px;
    margin: 8px 0 14px;
    font-size: 0.84rem;
    color: #C8CCEC;
    line-height: 1.7;
}
.finding strong { color: var(--gold); }

.tlight {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin: 6px 0;
}
.dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-g { background: var(--mint);   box-shadow: 0 0 10px rgba(0,229,195,0.5); }
.dot-a { background: var(--gold);   box-shadow: 0 0 10px rgba(245,200,66,0.5); }
.dot-r { background: var(--coral);  box-shadow: 0 0 10px rgba(255,107,107,0.5); }
.tlight-txt { font-size: 0.86rem; font-weight: 600; color: var(--white); flex: 1; }
.tlight-sub { font-size: 0.75rem; color: var(--muted); }

.badge {
    display:inline-block;
    border-radius:6px;
    padding:3px 10px;
    font-size:0.75rem;
    font-weight:700;
    margin:2px 3px;
}
.badge-g { background:rgba(0,229,195,0.15); color:var(--mint); }
.badge-a { background:rgba(245,200,66,0.15); color:var(--gold); }
.badge-r { background:rgba(255,107,107,0.15); color:var(--coral); }
.badge-v { background:rgba(168,85,247,0.15); color:var(--violet); }

.pred-hero {
    background: linear-gradient(135deg, #0B0F2E 0%, #1a0f35 50%, #0f1a35 100%);
    border: 1px solid rgba(245,200,66,0.3);
    border-radius: 20px;
    padding: 36px 28px;
    text-align: center;
    box-shadow: 0 0 60px rgba(245,200,66,0.08);
    position: relative;
    overflow: hidden;
}
.pred-hero::before {
    content: "";
    position: absolute;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(245,200,66,0.12) 0%, transparent 70%);
    top: -90px; left: 50%;
    transform: translateX(-50%);
    border-radius: 50%;
    pointer-events: none;
}
.ph-label {
    color: var(--muted);
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 12px;
}
.ph-value {
    font-family: 'Syne', sans-serif;
    font-size: 4.25rem;
    font-weight: 800;
    color: var(--gold);
    line-height: 1;
}
.ph-range {
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 8px;
}
.ph-verdict {
    font-size: 1.05rem;
    font-weight: 700;
    margin-top: 16px;
    color: var(--white);
}

section[data-testid="stSidebar"] { background: var(--navy2) !important; }
section[data-testid="stSidebar"] * { color: var(--white) !important; }

.js-plotly-plot .plotly .modebar { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# Helpers
# =========================================================
def canonical(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower().strip())


def safe_text(x, default="N/A"):
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    return str(x)


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.number)):
            return float(x)
        return float(str(x).replace("%", "").strip())
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def discover_versions():
    versions = []
    if not os.path.exists(MODEL_BASE_FOLDER):
        return versions

    for folder in os.listdir(MODEL_BASE_FOLDER):
        folder_path = os.path.join(MODEL_BASE_FOLDER, folder)
        registry_path = os.path.join(folder_path, "model_registry.json")

        if not os.path.isdir(folder_path):
            continue

        if not re.fullmatch(r"v\d+", folder):
            continue

        if not os.path.exists(registry_path):
            continue

        versions.append(folder)

    return sorted(versions, key=lambda x: int(x[1:]))


def get_active_version():
    if os.path.exists(ACTIVE_VERSION_FILE):
        try:
            with open(ACTIVE_VERSION_FILE, "r", encoding="utf-8") as f:
                value = f.read().strip()
                if value:
                    return value
        except Exception:
            pass

    versions = discover_versions()
    if versions:
        return versions[-1]

    return "v1"


def resolve_registry_path(version):
    candidates = [
        os.path.join(MODEL_BASE_FOLDER, version, "model_registry.json"),
        os.path.join(MODEL_BASE_FOLDER, version, "registry.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


@st.cache_data(ttl=20, show_spinner=False)
def load_data():
    engine = get_engine()
    try:
        return pd.read_sql_query("SELECT * FROM terms_enrollment", engine)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=20, show_spinner=False)
def load_dataset_versions():
    engine = get_engine()
    try:
        return pd.read_sql_query("SELECT * FROM dataset_versions", engine)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=20, show_spinner=False)
def load_model_history():
    engine = get_engine()
    try:
        return pd.read_sql_query("SELECT * FROM model_history", engine)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=20, show_spinner=False)
def load_registry(version):
    path = resolve_registry_path(version)
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    if isinstance(data, list):
        matches = [
            item for item in data
            if isinstance(item, dict) and canonical(item.get("version", "")) == canonical(version)
        ]
        if matches:
            return matches[-1]
        if data and isinstance(data[-1], dict):
            return data[-1]
        return {}

    if isinstance(data, dict):
        return data

    return {}


@st.cache_resource(show_spinner=False)
def load_model(version):
    path = os.path.join(MODEL_BASE_FOLDER, version, "best_model.joblib")
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def latest_row_dict(df):
    if df.empty:
        return {}
    if "term_start_date" in df.columns:
        temp = df.copy()
        temp["term_start_date"] = pd.to_datetime(temp["term_start_date"], errors="coerce")
        temp = temp.sort_values("term_start_date")
        return temp.iloc[-1].to_dict()
    return df.iloc[-1].to_dict()


def kpi_card(icon, value, label, sub="", color="#F5C842"):
    st.markdown(
        f"""
        <div class="kcard" style="--accent:{color}">
          <div class="ki">{icon}</div>
          <div class="kv">{value}</div>
          <div class="kl">{label}</div>
          <div class="ks">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_input_df(payload):
    return pd.DataFrame([{col: payload.get(col, 0) for col in FEATURE_COLUMNS}])


def predict_with_model(model, payload):
    if model is None:
        return None
    try:
        x = build_input_df(payload)
        pred = model.predict(x)[0]
        return round(float(pred), 2)
    except Exception:
        return None


def model_metrics_from_registry(registry, model_name):
    if not registry:
        return {}

    def norm(x):
        return str(x).lower().replace(" ", "").replace("_", "")

    model_name_n = norm(model_name)

    all_models = registry.get("all_models")
    if isinstance(all_models, dict) and len(all_models) > 0:
        for k, v in all_models.items():
            if norm(k) == model_name_n:
                return v

    best_model = registry.get("best_model")
    metrics = registry.get("metrics", {})
    if best_model and norm(best_model) == model_name_n:
        return metrics

    return {}


def registry_summary(registry, fallback_version):
    if not registry:
        return {
            "version": fallback_version,
            "best_model": "N/A",
            "trained_at": "N/A",
            "metrics": {},
            "all_models": {},
        }
    return {
        "version": registry.get("version", fallback_version),
        "best_model": registry.get("best_model", "N/A"),
        "trained_at": registry.get("trained_at", "N/A"),
        "metrics": registry.get("metrics", {}) or {},
        "all_models": registry.get("all_models", {}) or {},
    }


def extract_feature_importance(model):
    if model is None or not hasattr(model, "named_steps"):
        return None
    try:
        pre = model.named_steps.get("preprocessor")
        mdl = model.named_steps.get("model")
        if pre is None or mdl is None:
            return None

        if hasattr(pre, "get_feature_names_out") and hasattr(mdl, "feature_importances_"):
            feat_names = list(pre.get_feature_names_out())
            vals = mdl.feature_importances_
            return pd.DataFrame({"feature": feat_names, "importance": vals}).sort_values("importance", ascending=False)

        if hasattr(pre, "get_feature_names_out") and hasattr(mdl, "coef_"):
            feat_names = list(pre.get_feature_names_out())
            vals = np.abs(mdl.coef_)
            return pd.DataFrame({"feature": feat_names, "importance": vals}).sort_values("importance", ascending=False)
    except Exception:
        return None
    return None


def registry_to_rows(registry, version):
    if not registry:
        return pd.DataFrame(columns=["version", "model", "mae", "rmse", "r2", "mape", "best", "training_rows", "trained_at"])

    all_models = registry.get("all_models", {})
    rows = []
    for model_name, metrics in all_models.items():
        rows.append(
            {
                "version": version,
                "model": model_name,
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "r2": metrics.get("r2"),
                "mape": metrics.get("mape"),
                "best": model_name == registry.get("best_model"),
                "training_rows": registry.get("training_rows"),
                "trained_at": registry.get("trained_at"),
            }
        )
    return pd.DataFrame(rows)


# =========================================================
# Load data / registries / models
# =========================================================
df = load_data()
dataset_versions_df = load_dataset_versions()
model_history_df = load_model_history()

all_versions = discover_versions()
active_version = get_active_version()

active_registry_raw = load_registry(active_version)
active_registry = registry_summary(active_registry_raw, active_version)
active_model = load_model(active_version)

if df.empty:
    st.error("No data found. Run `python main.py` first so the dataset is created and loaded into SQLite.")
    st.stop()

if TARGET_COLUMN not in df.columns:
    st.error(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
    st.stop()

latest = latest_row_dict(df)

total_rows = len(df)
latest_batch_rows = int(dataset_versions_df["rows_generated"].iloc[-1]) if not dataset_versions_df.empty else total_rows
version_count = int(df["dataset_version"].nunique()) if "dataset_version" in df.columns else len(all_versions)
active_r2 = safe_float(active_registry["metrics"].get("r2"), None)
active_mae = safe_float(active_registry["metrics"].get("mae"), None)
active_rmse = safe_float(active_registry["metrics"].get("rmse"), None)
active_mape = safe_float(active_registry["metrics"].get("mape"), None)
active_accuracy = round(active_r2 * 100, 1) if active_r2 is not None else None
active_training_rows = safe_int(active_registry.get("training_rows"), None)

term_means = (
    df.groupby("term_label")[TARGET_COLUMN].mean().reindex(DISPLAY_TERM_ORDER).fillna(0)
    if {"term_label", TARGET_COLUMN}.issubset(df.columns)
    else pd.Series(dtype=float)
)
school_means = (
    df.groupby("school")[TARGET_COLUMN].mean().sort_values(ascending=False)
    if {"school", TARGET_COLUMN}.issubset(df.columns)
    else pd.Series(dtype=float)
)

dataset_version_counts = (
    df["dataset_version"].value_counts().sort_index()
    if "dataset_version" in df.columns
    else pd.Series(dtype=int)
)

reasons_totals = {}
for col in ["num_compulsory", "num_ge", "num_elective", "num_other"]:
    if col in df.columns:
        reasons_totals[col] = float(df[col].sum())

if "avg_remaining_credits" in df.columns:
    med_credits = df["avg_remaining_credits"].median()
    low_credit_mean = round(df[df["avg_remaining_credits"] <= med_credits][TARGET_COLUMN].mean(), 2)
    high_credit_mean = round(df[df["avg_remaining_credits"] > med_credits][TARGET_COLUMN].mean(), 2)
else:
    low_credit_mean = None
    high_credit_mean = None

latest_payload = {
    "school": latest.get("school", "VSST"),
    "term_label": latest.get("term_label", "independence"),
    "school_year1_population": safe_int(latest.get("school_year1_population"), 50),
    "school_year2_population": safe_int(latest.get("school_year2_population"), 45),
    "school_year3_population": safe_int(latest.get("school_year3_population"), 40),
    "school_year4_population": safe_int(latest.get("school_year4_population"), 35),
    "total_students_in_school": safe_int(latest.get("total_students_in_school"), 0),
    "avg_remaining_credits": float(latest.get("avg_remaining_credits", 20.0) or 20.0),
    "prev_term_enrollment": safe_int(latest.get("prev_term_enrollment"), 30),
    "prev2_term_enrollment": safe_int(latest.get("prev2_term_enrollment"), 25),
    "recent_trend": safe_int(latest.get("recent_trend"), 0),
}
active_model_prediction = predict_with_model(active_model, latest_payload)

# Build all model rows across every discovered version
all_model_rows = []
for version in all_versions:
    reg = load_registry(version)
    rows = registry_to_rows(reg, version)
    if not rows.empty:
        all_model_rows.append(rows)

all_version_models_df = pd.concat(all_model_rows, ignore_index=True) if all_model_rows else pd.DataFrame()

# =========================================================
# Hero
# =========================================================
st.markdown(
    f"""
<div class="hero">
  <div class="logo">EduCast AI · Campus Demand Intelligence</div>
  <h1>Turn Workshop Data into<br><span>Enrollment Intelligence</span></h1>
  <p>
    The dashboard reads the full database state, so the current total rows are always shown correctly.
    The latest generated batch and all versioned model results are also displayed separately.
  </p>
  <div class="hero-stats">
    <div class="hstat"><div class="hv">{total_rows}</div><div class="hl">Current DB Rows</div></div>
    <div class="hstat"><div class="hv">{latest_batch_rows}</div><div class="hl">Latest Batch Rows</div></div>
    <div class="hstat"><div class="hv">{version_count}</div><div class="hl">Dataset Versions</div></div>
    <div class="hstat"><div class="hv">{safe_text(active_version)}</div><div class="hl">Active Version</div></div>
    <div class="hstat"><div class="hv">{safe_text(active_registry["best_model"])}</div><div class="hl">Best Model</div></div>
    <div class="hstat"><div class="hv">{safe_text(active_accuracy if active_accuracy is not None else "N/A")}%</div><div class="hl">R² Accuracy</div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("EduCast AI")
    st.caption("Dashboard control panel")
    st.markdown("---")
    if st.button("Refresh dashboard"):
        st.cache_data.clear()
        st.cache_resource.clear()
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    st.markdown("### Current state")
    st.write(f"Current DB rows: **{total_rows}**")
    st.write(f"Latest batch rows: **{latest_batch_rows}**")
    st.write(f"Active version: **{safe_text(active_version)}**")
    st.write(f"Best model: **{safe_text(active_registry['best_model'])}**")
    st.write(f"Target: **{TARGET_COLUMN}**")
    st.markdown("---")
    st.info("Run `python main.py` to generate data, retrain, and refresh the active version.")

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Overview",
    "📊 Insights",
    "🔮 Predictor",
    "🤖 AI Performance",
    "🧾 Model Registry",
    "🛡️ Data Health",
    "📈 Version Comparison",
])

# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tab1:
    above_avg_pct = round((df[TARGET_COLUMN] >= df[TARGET_COLUMN].mean()).mean() * 100, 1)

    cols = st.columns(6)
    with cols[0]:
        kpi_card("🎓", f"{total_rows}", "Current DB Rows", "all versions combined", "#F5C842")
    with cols[1]:
        kpi_card("📦", f"{latest_batch_rows}", "Latest Batch Rows", "most recent generation", "#00E5C3")
    with cols[2]:
        kpi_card("🧩", f"{version_count}", "Dataset Versions", "tracked in DB", "#A855F7")
    with cols[3]:
        kpi_card("🤖", safe_text(active_registry["best_model"]), "Active Best Model", f"version {safe_text(active_version)}", "#38BDF8")
    with cols[4]:
        kpi_card("🎯", f"{active_accuracy if active_accuracy is not None else 'N/A'}%", "R² Accuracy", f"MAE {safe_text(active_mae)} | MAPE {safe_text(active_mape)}", "#FF6B6B")
    with cols[5]:
        kpi_card("⭐", f"{above_avg_pct}%", "Above Average", "rows above mean", "#FB923C")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.5])

    with left:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">🚦 Enrollment Health Check</div>
          <div class="panel-sub">Current state of the dataset</div>
        """, unsafe_allow_html=True)

        checks = [
            (total_rows >= 80, f"{total_rows} rows", "Enough rows for a meaningful training set"),
            (active_r2 is not None and active_r2 >= 0.80, f"R² {active_r2:.3f}" if active_r2 is not None else "R² N/A", "Good fit for demo and viva"),
            (active_mape is not None and active_mape < 100, f"MAPE {active_mape:.2f}%" if active_mape is not None else "MAPE N/A", "Lower is better"),
            (active_rmse is not None, f"RMSE {active_rmse:.3f}" if active_rmse is not None else "RMSE N/A", "Available for prediction range"),
            (active_training_rows is not None, f"Trained on {active_training_rows} rows" if active_training_rows is not None else "Training rows N/A", "Rows used for the active model"),
        ]

        for ok, txt, sub in checks:
            dot = "dot-g" if ok else "dot-a"
            st.markdown(
                f"""
                <div class="tlight">
                  <div class="dot {dot}"></div>
                  <div class="tlight-txt">{txt}</div>
                  <div class="tlight-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
          <div class="panel-title">📦 Enrollment Reasons</div>
          <div class="panel-sub">Compulsory / GE / Elective / Other distribution</div>
        """, unsafe_allow_html=True)

        if reasons_totals:
            reasons_df = pd.DataFrame({
                "Reason": [k.replace("num_", "").title() for k in reasons_totals.keys()],
                "Count": list(reasons_totals.values()),
            })
            fig = px.bar(
                reasons_df,
                x="Reason",
                y="Count",
                color="Count",
                color_continuous_scale="Plasma",
                text="Count",
            )
            fig.update_traces(marker_line_color="rgba(0,0,0,0)")
            fig.update_layout(**PLOTLY_TEMPLATE, showlegend=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No reason columns found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">📅 Enrollment Trend by Term</div>
          <div class="panel-sub">Seasonality across academic terms</div>
        """, unsafe_allow_html=True)

        if not term_means.empty:
            trend_df = term_means.reset_index()
            trend_df.columns = ["term_label", "avg_enrollment"]
            trend_df["term_label"] = trend_df["term_label"].map(TERM_LABELS).fillna(trend_df["term_label"])
            fig = px.line(trend_df, x="term_label", y="avg_enrollment", markers=True, color_discrete_sequence=["#F5C842"])
            fig.update_traces(line=dict(width=4), marker=dict(size=10))
            fig.update_layout(**PLOTLY_TEMPLATE, xaxis_title="", yaxis_title="Average Enrollment")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No term data available.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
          <div class="panel-title">🏫 Enrollment by School</div>
          <div class="panel-sub">Who contributes the most enrollments?</div>
        """, unsafe_allow_html=True)

        if not school_means.empty:
            school_df = school_means.reset_index()
            school_df.columns = ["school", "avg_enrollment"]
            fig = px.bar(
                school_df,
                x="school",
                y="avg_enrollment",
                color="avg_enrollment",
                color_continuous_scale="Plasma",
                text="avg_enrollment",
            )
            fig.update_traces(marker_line_color="rgba(0,0,0,0)")
            fig.update_layout(**PLOTLY_TEMPLATE, xaxis_title="", yaxis_title="Average Enrollment")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No school data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
      <div class="panel-title">🧠 Quick Findings</div>
      <div class="panel-sub">Plain-English summary from the dataset</div>
    """, unsafe_allow_html=True)

    best_term = term_means.idxmax() if not term_means.empty else None
    term_text = f"{TERM_LABELS.get(best_term, best_term)} term is strongest overall." if best_term is not None else "Term effect is not available."
    school_text = "VSST appears to carry compulsory enrollment strongly." if "VSST" in df["school"].unique() else "School distribution is fairly spread."
    credit_text = (
        f"Lower-credit rows average {low_credit_mean} while higher-credit rows average {high_credit_mean}."
        if low_credit_mean is not None and high_credit_mean is not None
        else "Credit-pressure effect is visible in the data."
    )

    st.markdown(f'<div class="finding">💡 <strong>Term effect:</strong> {term_text}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="finding">🏫 <strong>School effect:</strong> {school_text}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="finding">🎯 <strong>Credits effect:</strong> {credit_text}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="finding">📦 <strong>Latest batch:</strong> {latest_batch_rows} rows were generated in the most recent run, while the current DB state contains {total_rows} rows.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 2 — INSIGHTS
# =========================================================
with tab2:
    st.markdown("""
    <div class="panel">
      <div class="panel-title">🔑 What Drives Enrollment the Most?</div>
      <div class="panel-sub">Feature relationships with the target</div>
    """, unsafe_allow_html=True)

    insight_feats = [c for c in FEATURE_COLUMNS if c not in {"school", "term_label"} and c in df.columns]
    corr_vals, corr_labels = [], []

    for c in insight_feats:
        try:
            corr = abs(df[c].corr(df[TARGET_COLUMN]))
            if pd.notna(corr):
                corr_vals.append(corr)
                corr_labels.append(c)
        except Exception:
            pass

    if corr_vals:
        corr_df = pd.DataFrame({"Feature": corr_labels, "Impact": corr_vals}).sort_values("Impact", ascending=True)
        fig = px.bar(
            corr_df,
            x="Impact",
            y="Feature",
            orientation="h",
            color="Impact",
            color_continuous_scale="Plasma",
            text="Impact",
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_color="rgba(0,0,0,0)")
        fig.update_layout(**PLOTLY_TEMPLATE, showlegend=False, xaxis_title="Correlation with Enrollment", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        top_feat = corr_df.iloc[-1]["Feature"]
        st.markdown(f'<div class="finding">🏆 <strong>{top_feat}</strong> is the strongest numeric signal in the current dataset.</div>', unsafe_allow_html=True)
    else:
        st.info("No numeric relationships available to chart.")

    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">🎚️ Credits vs Enrollment</div>
          <div class="panel-sub">Remaining credits influence workshop demand</div>
        """, unsafe_allow_html=True)
        if "avg_remaining_credits" in df.columns:
            fig = px.scatter(
                df,
                x="avg_remaining_credits",
                y=TARGET_COLUMN,
                color=TARGET_COLUMN,
                color_continuous_scale="Plasma",
                opacity=0.7,
                trendline="lowess",
            )
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(**PLOTLY_TEMPLATE, showlegend=False, xaxis_title="Avg Remaining Credits", yaxis_title="Enrollment")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No credits column found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">⏱️ Previous Term Momentum</div>
          <div class="panel-sub">How current enrollment tracks with previous enrollment</div>
        """, unsafe_allow_html=True)
        if "prev_term_enrollment" in df.columns:
            fig = px.scatter(
                df,
                x="prev_term_enrollment",
                y=TARGET_COLUMN,
                color=TARGET_COLUMN,
                color_continuous_scale="Viridis",
                opacity=0.7,
                trendline="lowess",
            )
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(**PLOTLY_TEMPLATE, showlegend=False, xaxis_title="Previous Term Enrollment", yaxis_title="Enrollment")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No previous term feature found.")
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">🧊 Term-wise Average Enrollment</div>
          <div class="panel-sub">Seasonality across academic terms</div>
        """, unsafe_allow_html=True)
        if not term_means.empty:
            tdf = term_means.reset_index()
            tdf.columns = ["term_label", "avg_enrollment"]
            tdf["term_label"] = tdf["term_label"].map(TERM_LABELS).fillna(tdf["term_label"])
            fig = px.bar(tdf, x="term_label", y="avg_enrollment", color="avg_enrollment", color_continuous_scale="Plasma")
            fig.update_traces(marker_line_color="rgba(0,0,0,0)")
            fig.update_layout(**PLOTLY_TEMPLATE, showlegend=False, xaxis_title="", yaxis_title="Average Enrollment")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No term data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">🏫 School-wise Average Enrollment</div>
          <div class="panel-sub">Contribution by school</div>
        """, unsafe_allow_html=True)
        if not school_means.empty:
            sdf = school_means.reset_index()
            sdf.columns = ["school", "avg_enrollment"]
            fig = px.bar(sdf, x="school", y="avg_enrollment", color="avg_enrollment", color_continuous_scale="Viridis")
            fig.update_traces(marker_line_color="rgba(0,0,0,0)")
            fig.update_layout(**PLOTLY_TEMPLATE, showlegend=False, xaxis_title="", yaxis_title="Average Enrollment")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No school data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">💎 Reason Breakdown</div>
          <div class="panel-sub">Compulsory vs GE vs Elective vs Other</div>
        """, unsafe_allow_html=True)
        if reasons_totals:
            reason_df = pd.DataFrame({
                "Reason": [k.replace("num_", "").upper() for k in reasons_totals.keys()],
                "Count": list(reasons_totals.values()),
            })
            fig = px.bar(reason_df, x="Reason", y="Count", color="Count", color_continuous_scale="Plasma", text="Count")
            fig.update_traces(marker_line_color="rgba(0,0,0,0)")
            fig.update_layout(**PLOTLY_TEMPLATE, showlegend=False, xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No reason columns found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c6:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">🔥 Credits Brackets</div>
          <div class="panel-sub">Higher remaining credits often mean stronger participation</div>
        """, unsafe_allow_html=True)
        if "avg_remaining_credits" in df.columns:
            temp = df.copy()
            temp["credits_bin"] = pd.cut(
                temp["avg_remaining_credits"],
                bins=[-0.1, 3, 6, 10, 20, 40],
                labels=["0–3", "4–6", "7–10", "11–20", "21–40"],
            )
            heat = temp.groupby("credits_bin")[TARGET_COLUMN].mean().to_frame().reset_index()
            fig = px.bar(heat, x="credits_bin", y=TARGET_COLUMN, color=TARGET_COLUMN, color_continuous_scale="Plasma")
            fig.update_traces(marker_line_color="rgba(0,0,0,0)")
            fig.update_layout(**PLOTLY_TEMPLATE, showlegend=False, xaxis_title="Credits Bracket", yaxis_title="Avg Enrollment")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No credits data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
      <div class="panel-title">📷 EDA Artifacts</div>
      <div class="panel-sub">Plots generated by your pipeline</div>
    """, unsafe_allow_html=True)

    plots_dir = Path(PLOTS_FOLDER)
    pngs = sorted(list(plots_dir.glob("*.png"))) if plots_dir.exists() else []
    if pngs:
        show_count = min(3, len(pngs))
        cols = st.columns(show_count)
        for idx, p in enumerate(pngs[:show_count]):
            with cols[idx]:
                st.image(str(p), caption=p.stem.replace("_", " ").title(), use_column_width=True)
    else:
        st.info("No EDA plots found yet. Run `scripts/eda.py` or your pipeline.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 3 — PREDICTOR
# =========================================================
with tab3:
    left, right = st.columns([1, 1.15])

    schools = sorted(df["school"].dropna().unique().tolist()) if "school" in df.columns else ["VSST", "TSM", "JAGSoM", "VSOD", "VSOL"]
    if not schools:
        schools = ["VSST", "TSM", "JAGSoM", "VSOD", "VSOL"]

    with left:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">📝 Prediction Inputs</div>
          <div class="panel-sub">Use the latest DB row as a template or build a scenario manually</div>
        """, unsafe_allow_html=True)

        use_latest = st.checkbox("Use latest DB row as template", value=True)
        template = latest.copy() if (use_latest and latest) else {}

        school = st.selectbox(
            "School",
            schools,
            index=schools.index(template.get("school", schools[0])) if template.get("school") in schools else 0,
        )
        term = st.selectbox(
            "Term",
            DISPLAY_TERM_ORDER,
            index=DISPLAY_TERM_ORDER.index(template.get("term_label", "independence")) if template.get("term_label") in DISPLAY_TERM_ORDER else 0,
        )

        c1, c2 = st.columns(2)
        with c1:
            y1 = st.number_input("Year 1 population", min_value=0, value=safe_int(template.get("school_year1_population"), 50), step=1)
            y2 = st.number_input("Year 2 population", min_value=0, value=safe_int(template.get("school_year2_population"), 45), step=1)
            y3 = st.number_input("Year 3 population", min_value=0, value=safe_int(template.get("school_year3_population"), 40), step=1)
        with c2:
            y4 = st.number_input("Year 4 population", min_value=0, value=safe_int(template.get("school_year4_population"), 35), step=1)
            credits = st.number_input("Avg remaining credits", min_value=0.0, value=float(template.get("avg_remaining_credits", 20.0) or 20.0), step=0.25)
            prev1 = st.number_input("Prev term enrollment", min_value=0, value=safe_int(template.get("prev_term_enrollment"), 30), step=1)
            prev2 = st.number_input("Prev-2 term enrollment", min_value=0, value=safe_int(template.get("prev2_term_enrollment"), 25), step=1)

        total = y1 + y2 + y3 + y4
        recent_trend = int(prev1 - prev2)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="panel">
          <div class="panel-title">📦 Input Summary</div>
          <div class="panel-sub">Features sent to the model</div>
          <div class="finding"><strong>Total Students:</strong> {total}</div>
          <div class="finding"><strong>Recent Trend:</strong> {recent_trend}</div>
          <div class="finding"><strong>School:</strong> {safe_text(school)} · <strong>Term:</strong> {safe_text(TERM_LABELS.get(term, term))}</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        payload = {
            "school": school,
            "term_label": term,
            "school_year1_population": int(y1),
            "school_year2_population": int(y2),
            "school_year3_population": int(y3),
            "school_year4_population": int(y4),
            "total_students_in_school": int(total),
            "avg_remaining_credits": float(credits),
            "prev_term_enrollment": int(prev1),
            "prev2_term_enrollment": int(prev2),
            "recent_trend": int(recent_trend),
        }

        pred = predict_with_model(active_model, payload)
        rmse = active_rmse
        lower = max(0, pred - rmse) if (pred is not None and rmse is not None) else None
        upper = pred + rmse if (pred is not None and rmse is not None) else None

        percentile = None
        if pred is not None:
            try:
                percentile = round((df[TARGET_COLUMN] < pred).mean() * 100, 1)
            except Exception:
                percentile = None

        if pred is not None:
            if pred >= df[TARGET_COLUMN].mean() * 1.1:
                verdict = "🚀 Strong demand expected"
                verdict_color = "#00E5C3"
            elif pred >= df[TARGET_COLUMN].mean() * 0.9:
                verdict = "📈 Near average demand"
                verdict_color = "#F5C842"
            else:
                verdict = "⚠️ Lower demand expected"
                verdict_color = "#FF6B6B"
        else:
            verdict = "Prediction unavailable"
            verdict_color = "#FF6B6B"

        st.markdown(f"""
        <div class="pred-hero">
          <div class="ph-label">Projected Enrollment</div>
          <div class="ph-value">{safe_text(pred, "N/A")}</div>
          <div class="ph-range">
            {f"Range: {lower:.1f} – {upper:.1f} (±RMSE)" if lower is not None and upper is not None else "Range unavailable"}
          </div>
          <div class="ph-verdict" style="color:{verdict_color}">{verdict}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
          <div class="panel-title">🧠 AI Insight</div>
          <div class="panel-sub">Plain-English explanation from the business logic</div>
        """, unsafe_allow_html=True)

        if pred is not None:
            lines = []
            if school == "VSST":
                lines.append("VSST has compulsory workshop participation, so baseline enrollment stays high.")
            else:
                lines.append("Non-VSST schools rely more on elective and GE demand.")
            if term == "independence":
                lines.append("Independence term usually has the strongest demand.")
            elif term == "colors":
                lines.append("Colors term usually sees the weakest demand.")
            if credits <= 6:
                lines.append("Low remaining credits are suppressing demand.")
            elif credits >= 20:
                lines.append("High remaining credits support stronger participation.")
            if recent_trend > 0:
                lines.append("Recent momentum is positive.")
            elif recent_trend < 0:
                lines.append("Recent momentum is negative.")

            st.markdown(
                "<div class='finding'>" + " ".join([f"<strong>{x}</strong>" for x in lines]) + "</div>",
                unsafe_allow_html=True,
            )
            if percentile is not None:
                st.markdown(
                    f'<div class="finding">📊 This forecast is around the <strong>{percentile}th percentile</strong> of historical enrollment.</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning("Prediction failed. Check the active model and feature names.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
          <div class="panel-title">📉 Scenario Controls</div>
          <div class="panel-sub">Sanity-check how the forecast changes</div>
        """, unsafe_allow_html=True)

        credit_scale = st.slider("Credits scenario", 0.0, 40.0, float(credits), 0.25)
        prev_shift = st.slider("Previous-term shift", -50, 50, 0, 1)
        if st.button("Recalculate scenario"):
            scenario = payload.copy()
            scenario["avg_remaining_credits"] = float(credit_scale)
            scenario["prev_term_enrollment"] = max(0, int(prev1 + prev_shift))
            scenario["recent_trend"] = int(scenario["prev_term_enrollment"] - prev2)
            scenario_pred = predict_with_model(active_model, scenario)
            st.success(f"Scenario prediction: {safe_text(scenario_pred)}")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 4 — AI PERFORMANCE
# =========================================================
with tab4:
    st.markdown("""
    <div class="panel">
      <div class="panel-title">🤖 Model Comparison</div>
      <div class="panel-sub">All model entries from every discovered version are shown here</div>
    """, unsafe_allow_html=True)

    if all_version_models_df.empty:
        st.warning("No model registry found for the available versions.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        perf_df = all_version_models_df.copy()
        perf_df["label"] = perf_df["version"] + " | " + perf_df["model"]
        perf_df["best"] = perf_df["best"].map(lambda x: "Yes" if x else "No")

        show_cols = st.columns(min(4, len(perf_df)))
        for idx, (_, row) in enumerate(perf_df.iterrows()):
            col = show_cols[idx % len(show_cols)]
            is_best = row["best"] == "Yes"
            box_style = (
                "background:linear-gradient(135deg,#1E2347,#141837);"
                "border:1px solid rgba(245,200,66,0.5);box-shadow:0 0 30px rgba(245,200,66,0.12)"
                if is_best else
                "background:var(--navy2);border:1px solid rgba(255,255,255,0.08);"
            )
            col.markdown(
                f"""
                <div style="{box_style};border-radius:16px;padding:18px 14px;text-align:center;margin-bottom:12px;">
                  {('<div style="color:var(--gold);font-size:0.7rem;font-weight:800;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px;">⭐ BEST MODEL</div>') if is_best else '<div style="height:24px;margin-bottom:10px;"></div>'}
                  <div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#F0F2FF;margin-bottom:8px;">{safe_text(row["label"])}</div>
                  <div style="font-family:JetBrains Mono,monospace;font-size:1.9rem;font-weight:600;color:{'var(--gold)' if is_best else 'var(--muted)'};">{round(float(row['r2'])*100,1) if pd.notna(row['r2']) else 'N/A'}%</div>
                  <div style="font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin:4px 0 8px;">R² Accuracy</div>
                  <div style="font-size:0.8rem;color:#8B90B8;">MAE: {safe_text(row['mae'])} · RMSE: {safe_text(row['rmse'])} · MAPE: {safe_text(row['mape'])}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="panel">
              <div class="panel-title">📊 R² Across All Model Entries</div>
              <div class="panel-sub">Each row is version + model name</div>
            """, unsafe_allow_html=True)
            fig = px.bar(
                perf_df.sort_values("r2", ascending=True),
                x="label",
                y="r2",
                color="version",
                barmode="group",
                text=perf_df["r2"].round(3),
                color_discrete_sequence=["#A855F7", "#F5C842", "#00E5C3", "#38BDF8"],
            )
            fig.update_traces(marker_line_color="rgba(0,0,0,0)", textposition="outside")
            fig.update_layout(**PLOTLY_TEMPLATE, xaxis_title="", yaxis_title="R²")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="panel">
              <div class="panel-title">📉 Error Across All Model Entries</div>
              <div class="panel-sub">MAE / RMSE / MAPE across all versions</div>
            """, unsafe_allow_html=True)
            err_df = perf_df.melt(id_vars=["label", "version"], value_vars=["mae", "rmse", "mape"], var_name="metric", value_name="value")
            fig = px.bar(
                err_df,
                x="label",
                y="value",
                color="metric",
                barmode="group",
                color_discrete_sequence=["#00E5C3", "#F5C842", "#FF6B6B"],
            )
            fig.update_layout(**PLOTLY_TEMPLATE, xaxis_title="", yaxis_title="Error")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
          <div class="panel-title">🎯 Trade-off Analysis</div>
          <div class="panel-sub">Lower MAE, higher R², and manageable RMSE is ideal</div>
        """, unsafe_allow_html=True)
        fig = px.scatter(perf_df, x="mae", y="r2", size="rmse", color="label", hover_name="label")
        fig.update_layout(**PLOTLY_TEMPLATE, xaxis_title="MAE", yaxis_title="R²")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
          <div class="panel-title">🏅 Active Model Details</div>
          <div class="panel-sub">The currently deployed version and its metrics</div>
        """, unsafe_allow_html=True)
        st.json(
            {
                "active_version": active_version,
                "best_model": active_registry.get("best_model", "N/A"),
                "metrics": active_registry.get("metrics", {}),
                "trained_at": active_registry.get("trained_at", "N/A"),
                "training_rows": active_registry.get("training_rows", None),
                "current_db_rows": total_rows,
                "latest_batch_rows": latest_batch_rows,
            }
        )
        st.markdown("</div>", unsafe_allow_html=True)

    feat_imp_df = extract_feature_importance(active_model)
    if feat_imp_df is not None and not feat_imp_df.empty:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">🏅 Feature Influence</div>
          <div class="panel-sub">How the active saved pipeline sees your data</div>
        """, unsafe_allow_html=True)
        feat_plot = feat_imp_df.head(15).sort_values("importance")
        fig = px.bar(
            feat_plot,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Plasma",
        )
        fig.update_traces(marker_line_color="rgba(0,0,0,0)")
        fig.update_layout(**PLOTLY_TEMPLATE, xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 5 — MODEL REGISTRY
# =========================================================
with tab5:
    st.markdown("""
    <div class="panel">
      <div class="panel-title">🧾 Model Registry</div>
      <div class="panel-sub">Active version, historical versions, and current deployment state</div>
    """, unsafe_allow_html=True)
    st.json(active_registry_raw if active_registry_raw else active_registry)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">Active Version File</div>
        """, unsafe_allow_html=True)
        st.code(ACTIVE_VERSION_FILE)
        st.write(f"Current value: **{safe_text(active_version)}**")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">Discovered Versions</div>
        """, unsafe_allow_html=True)
        if all_versions:
            st.write(", ".join(all_versions))
        else:
            st.write("No versions discovered yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    if not model_history_df.empty:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">Model History Table</div>
          <div class="panel-sub">Logged after evaluation runs</div>
        """, unsafe_allow_html=True)
        st.dataframe(model_history_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 6 — DATA HEALTH
# =========================================================
with tab6:
    missing = int(df.isnull().sum().sum())
    dups = int(df.duplicated().sum())
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    outliers = 0

    for c in numeric_cols:
        series = df[c].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers += int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())

    checks = [
        ("📋", "Missing Data", f"{missing} missing values", missing == 0, "All records are complete" if missing == 0 else "Some cells need attention"),
        ("🔁", "Duplicate Rows", f"{dups} duplicates", dups == 0, "No duplicates found" if dups == 0 else "Duplicates should be reviewed"),
        ("📦", "Current DB Size", f"{total_rows} rows", total_rows >= 80, "Enough rows for a meaningful model"),
        ("🎯", "Outliers", f"{outliers} detected", outliers < 50, "Outliers look manageable" if outliers < 50 else "Review data quality"),
        ("🤖", "AI Readiness", f"{active_accuracy if active_accuracy is not None else 'N/A'}%", active_accuracy is not None and active_accuracy >= 80, "Good enough for demo" if active_accuracy is not None and active_accuracy >= 80 else "Model should be retrained"),
    ]

    st.markdown("""
    <div class="panel">
      <div class="panel-title">🛡️ Data Quality Report</div>
      <div class="panel-sub">A clean dataset is the foundation of reliable predictions</div>
    """, unsafe_allow_html=True)

    for icon, title, value, ok, explanation in checks:
        dot = "dot-g" if ok else "dot-a"
        badge = '<span class="badge badge-g">✓ Good</span>' if ok else '<span class="badge badge-a">⚠ Review</span>'
        st.markdown(
            f"""
            <div class="tlight">
              <div style="font-size:1.3rem;">{icon}</div>
              <div style="flex:1;">
                <div class="tlight-txt">{title} — <span style="color:var(--gold);font-family:JetBrains Mono,monospace;font-size:0.85rem;">{value}</span> {badge}</div>
                <div class="tlight-sub">{explanation}</div>
              </div>
              <div class="dot {dot}"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
      <div class="panel-title">📦 Dataset Version Distribution</div>
      <div class="panel-sub">Shows how the current DB rows are split across generated versions</div>
    """, unsafe_allow_html=True)

    if not dataset_version_counts.empty:
        version_df = dataset_version_counts.reset_index()
        version_df.columns = ["dataset_version", "rows"]
        fig = px.bar(version_df, x="dataset_version", y="rows", color="rows", color_continuous_scale="Plasma", text="rows")
        fig.update_traces(marker_line_color="rgba(0,0,0,0)")
        fig.update_layout(**PLOTLY_TEMPLATE, xaxis_title="", yaxis_title="Rows")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No dataset_version column found.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
      <div class="panel-title">📏 Range Validation</div>
      <div class="panel-sub">Checking values against expected ranges</div>
    """, unsafe_allow_html=True)

    expected = {
        "school_year1_population": (0, 500),
        "school_year2_population": (0, 500),
        "school_year3_population": (0, 500),
        "school_year4_population": (0, 500),
        "total_students_in_school": (0, 5000),
        "avg_remaining_credits": (0, 40),
        "prev_term_enrollment": (0, 500),
        "prev2_term_enrollment": (0, 500),
        "recent_trend": (-500, 500),
        "num_compulsory": (0, 5000),
        "num_ge": (0, 5000),
        "num_elective": (0, 5000),
        "num_other": (0, 5000),
        TARGET_COLUMN: (0, 5000),
    }

    for col_name, (lo, hi) in expected.items():
        if col_name not in df.columns:
            continue
        mn = df[col_name].min()
        mx = df[col_name].max()
        ok = mn >= lo and mx <= hi
        badge = '<span class="badge badge-g">✓ Valid</span>' if ok else '<span class="badge badge-a">⚠ Check</span>'
        st.markdown(
            f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:0.85rem;gap:12px;">
              <span style="font-weight:600;color:#F0F2FF;width:240px;">{col_name}</span>
              <span style="color:var(--muted);">Found: <strong style="color:var(--gold);font-family:JetBrains Mono,monospace;">{mn:.1f} – {mx:.1f}</strong></span>
              <span style="color:var(--muted);">Expected: <strong style="color:#8B90B8;">{lo} – {hi}</strong></span>
              {badge}
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
      <div class="panel-title">📊 Distribution Overview</div>
      <div class="panel-sub">Visual sanity check of major numeric features</div>
    """, unsafe_allow_html=True)

    vis_cols = [
        c for c in [
            "school_year1_population",
            "avg_remaining_credits",
            "prev_term_enrollment",
            "recent_trend",
            TARGET_COLUMN,
        ] if c in df.columns
    ]

    if vis_cols:
        ncols = 2
        nrows = int(np.ceil(len(vis_cols) / ncols))
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[c.replace("_", " ").title() for c in vis_cols],
            horizontal_spacing=0.10,
            vertical_spacing=0.18,
        )
        for idx, col_name in enumerate(vis_cols):
            r = idx // ncols + 1
            c = idx % ncols + 1
            fig.add_trace(
                go.Histogram(
                    x=df[col_name],
                    nbinsx=20,
                    marker_color=COLORWAY[idx % len(COLORWAY)],
                    opacity=0.85,
                    showlegend=False,
                    marker_line_color="rgba(0,0,0,0)",
                ),
                row=r,
                col=c,
            )
        fig.update_layout(**PLOTLY_TEMPLATE, height=650)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric features found to plot.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 7 — VERSION COMPARISON
# =========================================================
with tab7:
    st.markdown("""
    <div class="panel">
      <div class="panel-title">📈 Version Comparison</div>
      <div class="panel-sub">Every version found in the models folder is compared here dynamically</div>
    """, unsafe_allow_html=True)

    if all_version_models_df.empty:
        st.warning("No version registries found yet. Run `python main.py` to create v1, v2, v3... registries.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        comp_df = all_version_models_df.copy()
        comp_df["version_model"] = comp_df["version"] + " | " + comp_df["model"]
        comp_df["best"] = comp_df["best"].map(lambda x: "Yes" if x else "No")

        # Summary cards
        total_versions = len(all_versions)
        total_models = len(comp_df)
        best_row = comp_df.sort_values(["version", "best"], ascending=[False, False]).iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("🧩", f"{total_versions}", "Versions Found", "dynamic model folders", "#A855F7")
        with c2:
            kpi_card("🤖", f"{total_models}", "Total Model Entries", "4 per version", "#00E5C3")
        with c3:
            best_model_label = safe_text(best_row["version_model"])
            kpi_card("🏆", best_model_label, "Latest Best Entry", "best in the most recent version", "#F5C842")
        with c4:
            kpi_card("🎯", f"{active_accuracy if active_accuracy is not None else 'N/A'}%", "Active R²", f"active version: {safe_text(active_version)}", "#38BDF8")

        st.markdown("<br>", unsafe_allow_html=True)

        # Full table of all versions + all models
        st.markdown("""
        <div class="panel">
          <div class="panel-title">📋 Full Registry Comparison Table</div>
          <div class="panel-sub">All models from all discovered versions are listed here</div>
        """, unsafe_allow_html=True)
        display_cols = [
            "version",
            "model",
            "best",
            "training_rows",
            "mae",
            "rmse",
            "r2",
            "mape",
            "trained_at",
        ]
        st.dataframe(comp_df[display_cols], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # R2 chart by version
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="panel">
              <div class="panel-title">R² by Version</div>
              <div class="panel-sub">Comparison across all versions</div>
            """, unsafe_allow_html=True)
            fig = px.bar(
                comp_df.sort_values(["version", "r2"]),
                x="version_model",
                y="r2",
                color="version",
                text=comp_df["r2"].round(3),
                color_discrete_sequence=["#A855F7", "#F5C842", "#00E5C3", "#38BDF8"],
            )
            fig.update_traces(marker_line_color="rgba(0,0,0,0)", textposition="outside")
            fig.update_layout(**PLOTLY_TEMPLATE, xaxis_title="", yaxis_title="R²")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="panel">
              <div class="panel-title">MAE by Version</div>
              <div class="panel-sub">Lower is better</div>
            """, unsafe_allow_html=True)
            fig = px.bar(
                comp_df.sort_values(["version", "mae"]),
                x="version_model",
                y="mae",
                color="version",
                text=comp_df["mae"].round(3),
                color_discrete_sequence=["#A855F7", "#F5C842", "#00E5C3", "#38BDF8"],
            )
            fig.update_traces(marker_line_color="rgba(0,0,0,0)", textposition="outside")
            fig.update_layout(**PLOTLY_TEMPLATE, xaxis_title="", yaxis_title="MAE")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
          <div class="panel-title">🔥 Heatmap of Metric Changes</div>
          <div class="panel-sub">Each row is a version-model pair</div>
        """, unsafe_allow_html=True)

        heat_df = comp_df[["version_model", "mae", "rmse", "r2", "mape"]].set_index("version_model")
        fig = px.imshow(
            heat_df,
            aspect="auto",
            color_continuous_scale="RdYlGn",
            text_auto=".3f",
        )
        fig.update_layout(**PLOTLY_TEMPLATE, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="finding">
            Active version is <strong>{safe_text(active_version)}</strong>. The dashboard compares every discovered registry folder dynamically, so when you run the pipeline again it will automatically include the new version and its four models.
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# Footer
# =========================================================
st.caption(
    f"Active version: {safe_text(active_version)} | Current DB rows: {total_rows} | Latest batch rows: {latest_batch_rows}"
)

