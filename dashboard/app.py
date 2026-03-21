# dashboard/app.py
import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =========================================================
# Project root + config
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

try:
    from project_config import (
        BASE_DIR,
        DATASET_FILE,
        FEATURE_COLUMNS as CONFIG_FEATURE_COLUMNS,
        PLOTS_FOLDER,
        TARGET_COLUMN as CONFIG_TARGET_COLUMN,
        MODEL_BASE_FOLDER,
        ACTIVE_VERSION_FILE,
        MODEL_VERSION,
    )
except Exception:
    from project_config import (
        BASE_DIR,
        DATASET_FILE,
        PLOTS_FOLDER,
        FEATURE_COLUMNS as CONFIG_FEATURE_COLUMNS,
        TARGET_COLUMN as CONFIG_TARGET_COLUMN,
    )
    MODEL_BASE_FOLDER = os.path.join(BASE_DIR, "models")
    ACTIVE_VERSION_FILE = os.path.join(MODEL_BASE_FOLDER, "active_version.txt")
    MODEL_VERSION = "v2"

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
# Correct dashboard columns
# =========================================================
TARGET_COLUMN = "enrollment_this_term"
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
    "recent_trend",
]

TERM_ORDER = ["independence", "festivals", "republic", "colors"]
TERM_LABELS = {
    "independence": "Independence",
    "festivals": "Festivals",
    "republic": "Republic",
    "colors": "Colors",
}

C = ["#F5C842", "#00E5C3", "#FF6B6B", "#A855F7", "#38BDF8", "#FB923C", "#4ADE80"]

PL = dict(
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
    colorway=C,
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
    --navy3:  #1E2347;
    --gold:   #F5C842;
    --gold2:  #FFD95A;
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

.placeiq-hero {
    background: linear-gradient(135deg, #0B0F2E 0%, #141837 40%, #1a1040 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 48px 52px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.placeiq-hero::before {
    content: "";
    position: absolute;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(245,200,66,0.08) 0%, transparent 65%);
    top: -200px; right: -100px;
    border-radius: 50%;
    pointer-events: none;
}
.placeiq-hero::after {
    content: "";
    position: absolute;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,229,195,0.07) 0%, transparent 65%);
    bottom: -100px; left: 100px;
    border-radius: 50%;
    pointer-events: none;
}
.piq-logo {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.25em;
    color: var(--gold);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.piq-logo::before {
    content: "";
    display: inline-block;
    width: 28px; height: 2px;
    background: var(--gold);
}
.placeiq-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 12px;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.placeiq-hero h1 span { color: var(--gold); }
.placeiq-hero p {
    color: var(--muted);
    font-size: 1rem;
    margin: 0 0 32px;
    font-weight: 400;
    max-width: 720px;
}
.hero-stats {
    display: flex;
    gap: 40px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
    flex-wrap: wrap;
}
.hstat .hv {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
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
    padding: 20px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
    min-width: 170px;
}
.kcard:hover { border-color: var(--gold); transform: translateY(-2px); }
.kcard::before {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, var(--gold));
}
.kcard .ki { font-size: 1.4rem; margin-bottom: 10px; }
.kcard .kv {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.7rem;
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
    padding: 28px 30px;
    margin-bottom: 18px;
}
.panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 4px;
    letter-spacing: -0.01em;
}
.panel-sub {
    font-size: 0.82rem;
    color: var(--muted);
    margin: 0 0 20px;
    font-weight: 400;
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
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin: 6px 0;
}
.dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
.dot-g { background: var(--mint);   box-shadow: 0 0 10px rgba(0,229,195,0.5); }
.dot-a { background: var(--gold);   box-shadow: 0 0 10px rgba(245,200,66,0.5); }
.dot-r { background: var(--coral);  box-shadow: 0 0 10px rgba(255,107,107,0.5); }
.tlight-txt { font-size: 0.86rem; font-weight: 600; color: var(--white); flex: 1; }
.tlight-sub { font-size: 0.75rem; color: var(--muted); }

.badge { display:inline-block; border-radius:6px; padding:3px 10px; font-size:0.75rem; font-weight:700; margin:2px 3px; }
.badge-g { background:rgba(0,229,195,0.15); color:var(--mint); }
.badge-a { background:rgba(245,200,66,0.15); color:var(--gold); }
.badge-r { background:rgba(255,107,107,0.15); color:var(--coral); }
.badge-v { background:rgba(168,85,247,0.15); color:var(--violet); }

.pred-hero {
    background: linear-gradient(135deg, #0B0F2E 0%, #1a0f35 50%, #0f1a35 100%);
    border: 1px solid rgba(245,200,66,0.3);
    border-radius: 20px;
    padding: 40px 32px;
    text-align: center;
    box-shadow: 0 0 60px rgba(245,200,66,0.08);
    position: relative;
    overflow: hidden;
}
.pred-hero::before {
    content: "";
    position: absolute;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(245,200,66,0.12) 0%, transparent 70%);
    top: -100px; left: 50%;
    transform: translateX(-50%);
    border-radius: 50%;
    pointer-events: none;
}
.ph-label { color: var(--muted); font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 12px; }
.ph-value { font-family: 'Syne', sans-serif; font-size: 4.5rem; font-weight: 800; color: var(--gold); line-height: 1; }
.ph-range { color: var(--muted); font-size: 0.85rem; margin-top: 8px; }
.ph-verdict { font-size: 1.1rem; font-weight: 700; margin-top: 16px; color: var(--white); }

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
def get_active_version():
    if os.path.exists(ACTIVE_VERSION_FILE):
        try:
            with open(ACTIVE_VERSION_FILE, "r") as f:
                v = f.read().strip()
                if v:
                    return v
        except Exception:
            pass
    return MODEL_VERSION or "v2"


@st.cache_data(ttl=20, show_spinner=False)
def load_data():
    engine = get_engine()
    try:
        df = pd.read_sql("SELECT * FROM terms_enrollment", engine)
        if not df.empty:
            return df
    except Exception:
        pass

    if os.path.exists(DATASET_FILE):
        return pd.read_csv(DATASET_FILE)

    return pd.DataFrame()


@st.cache_data(ttl=20, show_spinner=False)
def load_registry_by_version(version):
    path = os.path.join(MODEL_BASE_FOLDER, version, "model_registry.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_resource(show_spinner=False)
def load_model_for_version(version):
    model_path = os.path.join(MODEL_BASE_FOLDER, version, "best_model.joblib")
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def latest_row_dict(df):
    if df.empty:
        return {}
    if "term_start_date" in df.columns:
        try:
            temp = df.copy()
            temp["term_start_date"] = pd.to_datetime(temp["term_start_date"], errors="coerce")
            temp = temp.sort_values("term_start_date")
            return temp.iloc[-1].to_dict()
        except Exception:
            pass
    return df.iloc[-1].to_dict()


def safe_percentile(series, value):
    try:
        return round((series < value).mean() * 100, 1)
    except Exception:
        return None


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


def register_text(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_input_df(payload):
    data = {c: [payload.get(c, 0)] for c in FEATURE_COLUMNS}
    return pd.DataFrame(data)


def predict_with_model(model, payload):
    if model is None:
        return None
    try:
        input_df = build_input_df(payload)
        return round(float(model.predict(input_df)[0]), 2)
    except Exception:
        return None


def extract_feature_importance(model):
    if model is None or not hasattr(model, "named_steps"):
        return None

    try:
        pre = model.named_steps.get("preprocessor")
        mdl = model.named_steps.get("model")
        if pre is None or mdl is None:
            return None

        if hasattr(pre, "get_feature_names_out") and hasattr(mdl, "feature_importances_"):
            names = list(pre.get_feature_names_out())
            vals = mdl.feature_importances_
            return pd.DataFrame({"feature": names, "importance": vals}).sort_values("importance", ascending=False)

        if hasattr(pre, "get_feature_names_out") and hasattr(mdl, "coef_"):
            names = list(pre.get_feature_names_out())
            vals = np.abs(mdl.coef_)
            return pd.DataFrame({"feature": names, "importance": vals}).sort_values("importance", ascending=False)

    except Exception:
        return None

    return None


# =========================================================
# Load data / version / models
# =========================================================
df = load_data()
active_version = get_active_version()
active_registry = load_registry_by_version(active_version)
active_model = load_model_for_version(active_version)

v1_registry = load_registry_by_version("v1")
v2_registry = load_registry_by_version("v2")

if df.empty:
    st.error("No data found. Run `python main.py` first so the dataset is created and loaded into SQLite.")
    st.stop()

latest = latest_row_dict(df)

best_model_name = active_registry.get("best_model", "Unknown")
best_metrics = active_registry.get("metrics", {})
all_models = active_registry.get("all_models", {})

rows_count = len(df)
avg_enrollment = round(df[TARGET_COLUMN].mean(), 2) if TARGET_COLUMN in df.columns else 0
peak_enrollment = round(df[TARGET_COLUMN].max(), 2) if TARGET_COLUMN in df.columns else 0
accuracy = round((best_metrics.get("r2", 0) or 0) * 100, 1) if isinstance(best_metrics.get("r2", None), (int, float)) else None
mae_value = best_metrics.get("mae", None)
rmse_value = best_metrics.get("rmse", None)
mape_value = best_metrics.get("mape", None)

feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

term_means = (
    df.groupby("term_label")[TARGET_COLUMN].mean().reindex(TERM_ORDER).fillna(0)
    if {"term_label", TARGET_COLUMN}.issubset(df.columns)
    else pd.Series(dtype=float)
)
school_means = (
    df.groupby("school")[TARGET_COLUMN].mean().sort_values(ascending=False)
    if {"school", TARGET_COLUMN}.issubset(df.columns)
    else pd.Series(dtype=float)
)

reasons_totals = {}
for col in ["num_compulsory", "num_ge", "num_elective", "num_other"]:
    if col in df.columns:
        reasons_totals[col] = float(df[col].sum())

if "avg_remaining_credits" in df.columns and TARGET_COLUMN in df.columns:
    med_credits = df["avg_remaining_credits"].median()
    low_credit_median = round(df[df["avg_remaining_credits"] <= med_credits][TARGET_COLUMN].mean(), 2)
    high_credit_median = round(df[df["avg_remaining_credits"] > med_credits][TARGET_COLUMN].mean(), 2)
else:
    low_credit_median = None
    high_credit_median = None

active_model_prediction = None
if active_model is not None:
    payload = {
        "school": latest.get("school", "VSST"),
        "term_label": latest.get("term_label", "independence"),
        "school_year1_population": int(latest.get("school_year1_population", 0) or 0),
        "school_year2_population": int(latest.get("school_year2_population", 0) or 0),
        "school_year3_population": int(latest.get("school_year3_population", 0) or 0),
        "school_year4_population": int(latest.get("school_year4_population", 0) or 0),
        "total_students_in_school": int(latest.get("total_students_in_school", 0) or 0),
        "avg_remaining_credits": float(latest.get("avg_remaining_credits", 0) or 0),
        "prev_term_enrollment": int(latest.get("prev_term_enrollment", 0) or 0),
        "prev2_term_enrollment": int(latest.get("prev2_term_enrollment", 0) or 0),
        "recent_trend": int(latest.get("recent_trend", 0) or 0),
    }
    active_model_prediction = predict_with_model(active_model, payload)

# =========================================================
# Hero
# =========================================================
st.markdown(
    f"""
<div class="placeiq-hero">
  <div class="piq-logo">EduCast AI · Campus Demand Intelligence</div>
  <h1>Turn Workshop Data into<br><span>Enrollment Intelligence</span></h1>
  <p>
    Predict next-term engineering workshop enrollment, inspect academic behavior, compare models, and monitor dataset health —
    all in one premium dashboard.
  </p>
  <div class="hero-stats">
    <div class="hstat"><div class="hv">{rows_count}</div><div class="hl">Rows Analysed</div></div>
    <div class="hstat"><div class="hv">{avg_enrollment}</div><div class="hl">Avg Enrollment</div></div>
    <div class="hstat"><div class="hv">{peak_enrollment}</div><div class="hl">Peak Enrollment</div></div>
    <div class="hstat"><div class="hv">{accuracy if accuracy is not None else "—"}%</div><div class="hl">R² Accuracy</div></div>
    <div class="hstat"><div class="hv">{best_model_name}</div><div class="hl">Best Model</div></div>
    <div class="hstat"><div class="hv">{active_version}</div><div class="hl">Active Version</div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("EduCast AI")
    st.caption("Emergency premium dashboard")
    st.markdown("---")
    if st.button("Refresh dashboard"):
        st.cache_data.clear()
        st.cache_resource.clear()
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
    st.markdown("### Dataset")
    st.write(f"Rows: **{rows_count}**")
    st.write(f"Active version: **{active_version}**")
    st.write(f"Best model: **{best_model_name}**")
    st.write(f"Target: **{TARGET_COLUMN}**")
    st.markdown("---")
    st.info("Run `python main.py` to refresh data, train the latest version, and launch this dashboard.")

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
    "🆚 V1 vs V2",
])

# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tab1:
    above_avg_pct = round((df[TARGET_COLUMN] >= df[TARGET_COLUMN].mean()).mean() * 100, 1) if TARGET_COLUMN in df.columns else 0

    cols = st.columns(6)
    with cols[0]:
        kpi_card("🎓", f"{rows_count}", "Rows in Dataset", "records available", "#F5C842")
    with cols[1]:
        kpi_card("📈", f"{avg_enrollment}", "Average Enrollment", "this term", "#00E5C3")
    with cols[2]:
        kpi_card("🏆", f"{peak_enrollment}", "Highest Enrollment", "single row peak", "#A855F7")
    with cols[3]:
        kpi_card("🤖", best_model_name, "Best Model", "active production candidate", "#38BDF8")
    with cols[4]:
        kpi_card("🎯", f"{accuracy if accuracy is not None else '—'}%", "R² Accuracy", f"MAE {register_text(mae_value)} | MAPE {register_text(mape_value)}", "#FF6B6B")
    with cols[5]:
        kpi_card("⭐", f"{above_avg_pct}%", "Above Average", "rows above mean", "#FB923C")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.5])

    with c1:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">🚦 Enrollment Health Check</div>
              <div class="panel-sub">What the dataset says at a glance</div>
            """,
            unsafe_allow_html=True,
        )

        checks = [
            (avg_enrollment >= 20, f"Avg enrollment {avg_enrollment}", "Healthy if 20+"),
            (peak_enrollment >= avg_enrollment, f"Peak {peak_enrollment}", "Above average peak"),
            (mape_value is not None and mape_value < 100, f"MAPE {register_text(mape_value)}%", "Lower is better"),
            (best_metrics.get("r2", 0) >= 0.80 if isinstance(best_metrics.get("r2", None), (int, float)) else False, f"R² {register_text(best_metrics.get('r2'))}", "Strong fit"),
            (rows_count >= 80, f"{rows_count} rows", "Enough data for training"),
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

        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">📦 Enrollment Reasons</div>
              <div class="panel-sub">How the reasons for enrollment split across the dataset</div>
            """,
            unsafe_allow_html=True,
        )
        if reasons_totals:
            reasons_df = pd.DataFrame(
                {
                    "Reason": [k.replace("num_", "").title() for k in reasons_totals.keys()],
                    "Count": list(reasons_totals.values()),
                }
            )
            fig = px.bar(
                reasons_df,
                x="Reason",
                y="Count",
                color="Count",
                color_continuous_scale="Plasma",
                text="Count",
            )
            fig.update_traces(marker_line_color="rgba(0,0,0,0)")
            fig.update_layout(**PL, showlegend=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No reason columns found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">📅 Enrollment Trend by Term</div>
              <div class="panel-sub">Independence → Festivals → Republic → Colors</div>
            """,
            unsafe_allow_html=True,
        )
        if not term_means.empty:
            trend_df = term_means.reset_index()
            trend_df.columns = ["term_label", "avg_enrollment"]
            trend_df["term_label"] = trend_df["term_label"].map(TERM_LABELS).fillna(trend_df["term_label"])
            fig = px.line(trend_df, x="term_label", y="avg_enrollment", markers=True, color_discrete_sequence=["#F5C842"])
            fig.update_traces(line=dict(width=4), marker=dict(size=10))
            fig.update_layout(**PL, xaxis_title="", yaxis_title="Average Enrollment")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No term data available.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">🏫 Enrollment by School</div>
              <div class="panel-sub">Who contributes the most workshop enrollments?</div>
            """,
            unsafe_allow_html=True,
        )
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
            fig.update_layout(**PL, xaxis_title="", yaxis_title="Average Enrollment")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No school data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">🧠 Quick Findings</div>
          <div class="panel-sub">Translated from the dataset into plain English</div>
        """,
        unsafe_allow_html=True,
    )
    if not term_means.empty:
        top_term = term_means.idxmax()
        finding_1 = f"{TERM_LABELS.get(top_term, top_term)} term is strongest overall"
    else:
        finding_1 = "Term effect not available"

    finding_2 = "VSST contributes compulsory enrollments heavily" if "VSST" in df["school"].unique() else "School distribution is balanced across the dataset"
    finding_3 = (
        f"Higher remaining credits correlate with {high_credit_median} average enrollment vs {low_credit_median} at lower credits"
        if high_credit_median is not None
        else "Credit-pressure relationship is visible in the data"
    )

    st.markdown(f'<div class="finding">💡 <strong>Term effect:</strong> {finding_1}.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="finding">🏫 <strong>School effect:</strong> {finding_2}.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="finding">🎯 <strong>Credits effect:</strong> {finding_3}.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 2 — INSIGHTS
# =========================================================
with tab2:
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">🔑 What Drives Enrollment the Most?</div>
          <div class="panel-sub">Feature relationships with the target</div>
        """,
        unsafe_allow_html=True,
    )

    insight_feats = [c for c in feature_cols if c not in {"school", "term_label"}]
    corr_vals, corr_labels = [], []

    for c in insight_feats:
        if c in df.columns:
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
        fig.update_layout(**PL, showlegend=False, xaxis_title="Correlation with Enrollment")
        st.plotly_chart(fig, width="stretch")

        top_feat = corr_df.iloc[-1]["Feature"]
        st.markdown(f'<div class="finding">🏆 <strong>{top_feat}</strong> is the strongest numeric predictor in your current dataset.</div>', unsafe_allow_html=True)
    else:
        st.info("No numeric relationships available to chart.")

    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">🎚️ Credits vs Enrollment</div>
              <div class="panel-sub">Remaining credits influence workshop demand</div>
            """,
            unsafe_allow_html=True,
        )
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
            fig.update_layout(**PL, showlegend=False, xaxis_title="Avg Remaining Credits", yaxis_title="Enrollment")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No credits column found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">⏱️ Previous Term Momentum</div>
              <div class="panel-sub">How current enrollment tracks with past enrollment</div>
            """,
            unsafe_allow_html=True,
        )
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
            fig.update_layout(**PL, showlegend=False, xaxis_title="Previous Term Enrollment", yaxis_title="Enrollment")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No previous term feature found.")
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">🧊 Term-wise Average Enrollment</div>
              <div class="panel-sub">Seasonality across academic terms</div>
            """,
            unsafe_allow_html=True,
        )
        if not term_means.empty:
            tdf = term_means.reset_index()
            tdf.columns = ["term_label", "avg_enrollment"]
            tdf["term_label"] = tdf["term_label"].map(TERM_LABELS).fillna(tdf["term_label"])
            fig = px.bar(tdf, x="term_label", y="avg_enrollment", color="avg_enrollment", color_continuous_scale="Plasma")
            fig.update_traces(marker_line_color="rgba(0,0,0,0)")
            fig.update_layout(**PL, showlegend=False, xaxis_title="", yaxis_title="Average Enrollment")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No term data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">🏫 School-wise Average Enrollment</div>
              <div class="panel-sub">Contribution by school</div>
            """,
            unsafe_allow_html=True,
        )
        if not school_means.empty:
            sdf = school_means.reset_index()
            sdf.columns = ["school", "avg_enrollment"]
            fig = px.bar(sdf, x="school", y="avg_enrollment", color="avg_enrollment", color_continuous_scale="Viridis")
            fig.update_traces(marker_line_color="rgba(0,0,0,0)")
            fig.update_layout(**PL, showlegend=False, xaxis_title="", yaxis_title="Average Enrollment")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No school data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">💎 Reason Breakdown</div>
              <div class="panel-sub">Compulsory vs GE vs Elective vs Other</div>
            """,
            unsafe_allow_html=True,
        )
        if reasons_totals:
            reason_df = pd.DataFrame(
                {
                    "Reason": [k.replace("num_", "").upper() for k in reasons_totals.keys()],
                    "Count": list(reasons_totals.values()),
                }
            )
            fig = px.bar(reason_df, x="Reason", y="Count", color="Count", color_continuous_scale="Plasma", text="Count")
            fig.update_traces(marker_line_color="rgba(0,0,0,0)")
            fig.update_layout(**PL, showlegend=False, xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No reason columns found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c6:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">🔥 Credits Brackets</div>
              <div class="panel-sub">Higher remaining credits often mean stronger participation</div>
            """,
            unsafe_allow_html=True,
        )
        if "avg_remaining_credits" in df.columns:
            temp = df.copy()
            temp["credits_bin"] = pd.cut(
                temp["avg_remaining_credits"],
                bins=[-0.1, 3, 6, 10, 20, 40],
                labels=["0–3", "4–6", "7–10", "11–20", "21–40"],
            )
            heat = temp.groupby("credits_bin")[TARGET_COLUMN].mean().to_frame()
            fig = px.bar(heat.reset_index(), x="credits_bin", y=TARGET_COLUMN, color=TARGET_COLUMN, color_continuous_scale="Plasma")
            fig.update_traces(marker_line_color="rgba(0,0,0,0)")
            fig.update_layout(**PL, showlegend=False, xaxis_title="Credits Bracket", yaxis_title="Avg Enrollment")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No credits data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">📷 EDA Artifacts</div>
          <div class="panel-sub">Plots generated by your pipeline</div>
        """,
        unsafe_allow_html=True,
    )
    plots_dir = Path(PLOTS_FOLDER)
    pngs = sorted(list(plots_dir.glob("*.png"))) if plots_dir.exists() else []
    if pngs:
        cols = st.columns(min(3, len(pngs)))
        for idx, p in enumerate(pngs[:3]):
            with cols[idx % len(cols)]:
                st.image(str(p), caption=p.stem.replace("_", " ").title(), width="stretch")
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
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">📝 Student / School Input</div>
              <div class="panel-sub">Use the latest row as template or build a scenario manually</div>
            """,
            unsafe_allow_html=True,
        )

        use_latest = st.checkbox("Use latest DB row as template", value=True)
        template = latest.copy() if (use_latest and latest) else {}

        school = st.selectbox(
            "School",
            schools,
            index=schools.index(template.get("school", schools[0])) if template.get("school") in schools else 0,
        )
        term = st.selectbox(
            "Term",
            TERM_ORDER,
            index=TERM_ORDER.index(template.get("term_label", "independence")) if template.get("term_label") in TERM_ORDER else 0,
        )

        c1, c2 = st.columns(2)
        with c1:
            y1 = st.number_input("Year 1 population", min_value=0, value=int(template.get("school_year1_population", 50) or 50), step=1)
            y2 = st.number_input("Year 2 population", min_value=0, value=int(template.get("school_year2_population", 45) or 45), step=1)
            y3 = st.number_input("Year 3 population", min_value=0, value=int(template.get("school_year3_population", 40) or 40), step=1)
        with c2:
            y4 = st.number_input("Year 4 population", min_value=0, value=int(template.get("school_year4_population", 35) or 35), step=1)
            credits = st.number_input("Avg remaining credits", min_value=0.0, value=float(template.get("avg_remaining_credits", 20.0) or 20.0), step=0.25)
            prev1 = st.number_input("Prev term enrollment", min_value=0, value=int(template.get("prev_term_enrollment", 30) or 30), step=1)
            prev2 = st.number_input("Prev-2 term enrollment", min_value=0, value=int(template.get("prev2_term_enrollment", 25) or 25), step=1)

        total = y1 + y2 + y3 + y4
        recent_trend = int(prev1 - prev2)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="panel">
              <div class="panel-title">📦 Input Summary</div>
              <div class="panel-sub">These are the features used by your model</div>
              <div class="finding"><strong>Total Students:</strong> {total}</div>
              <div class="finding"><strong>Recent Trend:</strong> {recent_trend}</div>
              <div class="finding"><strong>School:</strong> {school} · <strong>Term:</strong> {TERM_LABELS.get(term, term)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        rmse = rmse_value if isinstance(rmse_value, (int, float)) else None
        lower = max(0, pred - rmse) if (pred is not None and rmse is not None) else None
        upper = pred + rmse if (pred is not None and rmse is not None) else None
        percentile = safe_percentile(df[TARGET_COLUMN], pred) if pred is not None else None

        if pred is not None:
            if pred >= avg_enrollment * 1.1:
                verdict = "🚀 Strong demand expected"
                verdict_color = "#00E5C3"
            elif pred >= avg_enrollment * 0.9:
                verdict = "📈 Near average demand"
                verdict_color = "#F5C842"
            else:
                verdict = "⚠️ Lower demand expected"
                verdict_color = "#FF6B6B"
        else:
            verdict = "Prediction unavailable"
            verdict_color = "#FF6B6B"

        st.markdown(
            f"""
            <div class="pred-hero">
              <div class="ph-label">Projected Next-Term Enrollment</div>
              <div class="ph-value">{register_text(pred) if pred is not None else "—"}</div>
              <div class="ph-range">
                {f"Range: {lower:.1f} – {upper:.1f} (±RMSE)" if lower is not None else "Range unavailable"}
              </div>
              <div class="ph-verdict" style="color:{verdict_color}">{verdict}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">🧠 AI Insight</div>
              <div class="panel-sub">A plain-English explanation from your business rules</div>
            """,
            unsafe_allow_html=True,
        )

        if pred is not None:
            insight_lines = []
            if school == "VSST":
                insight_lines.append("VSST has compulsory workshop participation, so baseline enrollment stays high.")
            else:
                insight_lines.append("Non-VSST schools rely on GE, elective, and exploration demand.")
            if term == "independence":
                insight_lines.append("Independence term usually has the strongest demand.")
            elif term == "colors":
                insight_lines.append("Colors term usually sees the weakest demand due to seasonal fatigue and fewer credits.")
            if credits <= 6:
                insight_lines.append("Low remaining credits are suppressing demand.")
            elif credits >= 20:
                insight_lines.append("High remaining credits support stronger participation.")
            if recent_trend > 0:
                insight_lines.append("Recent momentum is positive, which supports the forecast.")
            elif recent_trend < 0:
                insight_lines.append("Recent momentum is negative, which may soften the forecast.")

            st.markdown(
                "<div class='finding'>" + " ".join([f"<strong>{line}</strong>" for line in insight_lines]) + "</div>",
                unsafe_allow_html=True,
            )
            if percentile is not None:
                st.markdown(
                    f'<div class="finding">📊 This forecast is roughly <strong>{percentile}th percentile</strong> against historical enrollments.</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning("Prediction failed. Check the active model file and feature names.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">📉 Scenario Controls</div>
              <div class="panel-sub">Use this to sanity-check how the forecast changes</div>
            """,
            unsafe_allow_html=True,
        )

        credit_scale = st.slider("Credits scenario", 0.0, 40.0, float(credits), 0.25)
        prev_shift = st.slider("Previous-term shift", -50, 50, 0, 1)
        if st.button("Recalculate scenario"):
            scenario_payload = payload.copy()
            scenario_payload["avg_remaining_credits"] = float(credit_scale)
            scenario_payload["prev_term_enrollment"] = max(0, int(prev1 + prev_shift))
            scenario_payload["recent_trend"] = int(scenario_payload["prev_term_enrollment"] - prev2)
            scenario_pred = predict_with_model(active_model, scenario_payload)
            st.success(f"Scenario prediction: {register_text(scenario_pred)}")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 4 — AI PERFORMANCE
# =========================================================
with tab4:
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">🤖 Model Comparison</div>
          <div class="panel-sub">Lower error and higher R² means a better fit</div>
        """,
        unsafe_allow_html=True,
    )

    if all_models:
        comparison_df = pd.DataFrame([{"model": k, **v} for k, v in all_models.items()]).sort_values("r2", ascending=False)

        n = len(comparison_df)
        mc = st.columns(n)
        for idx, (_, row) in enumerate(comparison_df.iterrows()):
            is_best = row["model"] == best_model_name
            box_style = (
                "background:linear-gradient(135deg,#1E2347,#141837);"
                "border:1px solid rgba(245,200,66,0.5);box-shadow:0 0 30px rgba(245,200,66,0.12)"
                if is_best else
                "background:var(--navy2);border:1px solid rgba(255,255,255,0.08);"
            )
            mc[idx].markdown(
                f"""
                <div style="{box_style};border-radius:16px;padding:22px;text-align:center;margin-bottom:12px;">
                  {('<div style="color:var(--gold);font-size:0.7rem;font-weight:800;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px;">⭐ BEST MODEL</div>') if is_best else '<div style="height:24px;margin-bottom:10px;"></div>'}
                  <div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#F0F2FF;margin-bottom:14px;">{row['model']}</div>
                  <div style="font-family:JetBrains Mono,monospace;font-size:2.2rem;font-weight:600;color:{'var(--gold)' if is_best else 'var(--muted)'};">{round(row['r2']*100,1)}%</div>
                  <div style="font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin:4px 0 10px;">Accuracy</div>
                  <div style="font-size:0.83rem;color:#8B90B8;">MAE: {register_text(row.get('mae'))} · RMSE: {register_text(row.get('rmse'))} · MAPE: {register_text(row.get('mape'))}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                <div class="panel">
                  <div class="panel-title">📊 R² Comparison</div>
                  <div class="panel-sub">Model fit across all trained models</div>
                """,
                unsafe_allow_html=True,
            )
            fig = px.bar(
                comparison_df,
                x="model",
                y="r2",
                color="r2",
                color_continuous_scale="Plasma",
                text=comparison_df["r2"].round(3),
            )
            fig.update_traces(marker_line_color="rgba(0,0,0,0)", textposition="outside")
            fig.update_layout(**PL, xaxis_title="", yaxis_title="R²")
            st.plotly_chart(fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown(
                """
                <div class="panel">
                  <div class="panel-title">📉 Error Comparison</div>
                  <div class="panel-sub">MAE / RMSE / MAPE across models</div>
                """,
                unsafe_allow_html=True,
            )
            err_df = comparison_df.melt(id_vars=["model"], value_vars=["mae", "rmse", "mape"], var_name="metric", value_name="value")
            fig = px.bar(
                err_df,
                x="model",
                y="value",
                color="metric",
                barmode="group",
                color_discrete_sequence=["#00E5C3", "#F5C842", "#FF6B6B"],
            )
            fig.update_layout(**PL, xaxis_title="", yaxis_title="Error")
            st.plotly_chart(fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">🎯 Trade-off Analysis</div>
              <div class="panel-sub">Lower MAE, higher R², and manageable RMSE is the ideal sweet spot</div>
            """,
            unsafe_allow_html=True,
        )
        fig = px.scatter(comparison_df, x="mae", y="r2", size="rmse", color="model", hover_name="model", title="")
        fig.update_layout(**PL, xaxis_title="MAE", yaxis_title="R²")
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">🏅 Best Model Details</div>
              <div class="panel-sub">The registry's active production candidate</div>
            """,
            unsafe_allow_html=True,
        )
        st.json(
            {
                "best_model": best_model_name,
                "metrics": best_metrics,
                "trained_at": active_registry.get("trained_at"),
                "version": active_registry.get("version"),
                "active_version": active_version,
            }
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No model registry found for the active version.")

    feat_imp_df = extract_feature_importance(active_model)
    if feat_imp_df is not None and not feat_imp_df.empty:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">🏅 Feature Influence</div>
              <div class="panel-sub">How the saved pipeline sees your data</div>
            """,
            unsafe_allow_html=True,
        )
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
        fig.update_layout(**PL, xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 5 — MODEL REGISTRY
# =========================================================
with tab5:
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">🧾 Model Registry</div>
          <div class="panel-sub">Versioning and metrics for every trained model</div>
        """,
        unsafe_allow_html=True,
    )
    st.json(active_registry)
    st.markdown("</div>", unsafe_allow_html=True)

    if all_models:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">Registered Models</div>
              <div class="panel-sub">Active version models only</div>
            """,
            unsafe_allow_html=True,
        )
        for name, meta in all_models.items():
            cols = st.columns([3, 1, 1])
            cols[0].markdown(f"**{name}**")
            cols[1].write(f"R²: {meta.get('r2', '—')}")
            cols[2].write(f"MAE: {meta.get('mae', '—')}")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 6 — DATA HEALTH
# =========================================================
with tab6:
    missing = int(df.isnull().sum().sum())
    dups = int(df.duplicated().sum())
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    outliers = 0

    for c in numeric:
        if df[c].dropna().empty:
            continue
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        outliers += int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())

    checks = [
        ("📋", "Missing Data", f"{missing} missing values", missing == 0, "All records are complete" if missing == 0 else "Some cells need attention"),
        ("🔁", "Duplicate Rows", f"{dups} duplicates", dups == 0, "No duplicates found" if dups == 0 else "Duplicates should be reviewed"),
        ("📦", "Dataset Size", f"{rows_count} rows", rows_count >= 80, "Enough rows for a meaningful model"),
        ("🎯", "Outliers", f"{outliers} detected", outliers < 50, "Outliers look manageable" if outliers < 50 else "Review data entry quality"),
        ("🤖", "AI Readiness", f"{accuracy if accuracy is not None else '—'}%", accuracy is not None and accuracy >= 80, "Good enough for demo" if accuracy is not None and accuracy >= 80 else "Model should be retrained"),
    ]

    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">🛡️ Data Quality Report</div>
          <div class="panel-sub">A clean dataset is the foundation of reliable predictions</div>
        """,
        unsafe_allow_html=True,
    )
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

    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">📏 Range Validation</div>
          <div class="panel-sub">Checking values against expected ranges</div>
        """,
        unsafe_allow_html=True,
    )
    EXPECTED = {
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

    for col, (lo, hi) in EXPECTED.items():
        if col not in df.columns:
            continue
        mn = df[col].min()
        mx = df[col].max()
        ok = mn >= lo and mx <= hi
        badge = '<span class="badge badge-g">✓ Valid</span>' if ok else '<span class="badge badge-a">⚠ Check</span>'
        st.markdown(
            f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:0.85rem;">
              <span style="font-weight:600;color:#F0F2FF;width:240px;">{col}</span>
              <span style="color:var(--muted);">Found: <strong style="color:var(--gold);font-family:JetBrains Mono,monospace;">{mn:.1f} – {mx:.1f}</strong></span>
              <span style="color:var(--muted);">Expected: <strong style="color:#8B90B8;">{lo} – {hi}</strong></span>
              {badge}
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">📊 Distribution Overview</div>
          <div class="panel-sub">Visual sanity check of major numeric features</div>
        """,
        unsafe_allow_html=True,
    )
    vis_cols = [c for c in [
        "school_year1_population",
        "avg_remaining_credits",
        "prev_term_enrollment",
        "recent_trend",
        TARGET_COLUMN,
    ] if c in df.columns]

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
                    marker_color=C[idx % len(C)],
                    opacity=0.85,
                    showlegend=False,
                    marker_line_color="rgba(0,0,0,0)",
                ),
                row=r,
                col=c,
            )
        fig.update_layout(**PL, height=650)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No numeric features found to plot.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">📷 EDA Artifacts</div>
          <div class="panel-sub">Plots generated by your pipeline</div>
        """,
        unsafe_allow_html=True,
    )
    plots_dir = Path(PLOTS_FOLDER)
    pngs = sorted(list(plots_dir.glob("*.png"))) if plots_dir.exists() else []
    if pngs:
        cols = st.columns(min(3, len(pngs)))
        for idx, p in enumerate(pngs[:3]):
            with cols[idx % len(cols)]:
                st.image(str(p), caption=p.stem.replace("_", " ").title(), width="stretch")
    else:
        st.info("No EDA plots found yet. Run `scripts/eda.py` or your pipeline.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 7 — V1 VS V2
# =========================================================
with tab7:
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">🆚 Version Comparison</div>
          <div class="panel-sub">Compare the old model family and the new model family side by side</div>
        """,
        unsafe_allow_html=True,
    )

    if not v1_registry or not v2_registry:
        st.warning("Both `models/v1/model_registry.json` and `models/v2/model_registry.json` are needed for the comparison tab.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        model_names = ["LinearRegression", "Ridge", "RandomForest", "GradientBoosting"]
        rows = []
        for model_name in model_names:
            v1_m = v1_registry.get("all_models", {}).get(model_name, {})
            v2_m = v2_registry.get("all_models", {}).get(model_name, {})

            rows.append({
                "Model": model_name,
                "v1 R²": v1_m.get("r2"),
                "v2 R²": v2_m.get("r2"),
                "Δ R²": (v2_m.get("r2", 0) - v1_m.get("r2", 0)) if v1_m and v2_m else None,
                "v1 MAE": v1_m.get("mae"),
                "v2 MAE": v2_m.get("mae"),
                "Δ MAE": (v2_m.get("mae", 0) - v1_m.get("mae", 0)) if v1_m and v2_m else None,
                "v1 RMSE": v1_m.get("rmse"),
                "v2 RMSE": v2_m.get("rmse"),
                "Δ RMSE": (v2_m.get("rmse", 0) - v1_m.get("rmse", 0)) if v1_m and v2_m else None,
                "v1 MAPE": v1_m.get("mape"),
                "v2 MAPE": v2_m.get("mape"),
                "Δ MAPE": (v2_m.get("mape", 0) - v1_m.get("mape", 0)) if v1_m and v2_m else None,
            })

        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                comp_df.melt(id_vars="Model", value_vars=["v1 R²", "v2 R²"]),
                x="Model",
                y="value",
                color="variable",
                barmode="group",
                title="R²: v1 vs v2",
                color_discrete_sequence=["#A855F7", "#F5C842"],
            )
            fig.update_layout(**PL, xaxis_title="", yaxis_title="R²")
            st.plotly_chart(fig, width="stretch")

        with c2:
            fig = px.bar(
                comp_df.melt(id_vars="Model", value_vars=["v1 MAE", "v2 MAE"]),
                x="Model",
                y="value",
                color="variable",
                barmode="group",
                title="MAE: v1 vs v2",
                color_discrete_sequence=["#38BDF8", "#00E5C3"],
            )
            fig.update_layout(**PL, xaxis_title="", yaxis_title="MAE")
            st.plotly_chart(fig, width="stretch")

        st.markdown(
            f"""
            <div class="finding">
            Active version is <strong>{active_version}</strong>. The dashboard reads that version automatically.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)