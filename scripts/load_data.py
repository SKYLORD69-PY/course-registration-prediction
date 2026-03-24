# scripts/load_data.py

import os
import sys
import pandas as pd

# ==============================
# Fix import path
# ==============================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from database.db_connection import get_engine, get_connection
from database.generate_dataset import generate


# ==============================
# CREATE TABLES
# ==============================
def create_tables():
    conn = get_connection()

    sql_path = os.path.join(PROJECT_ROOT, "database", "create_tables.sql")

    with open(sql_path, "r") as f:
        sql_script = f.read()

    conn.executescript(sql_script)
    conn.commit()
    conn.close()

    print("✅ Database tables ready")


# ==============================
# LOAD DATASET (FIXED)
# ==============================
def load_dataset(
    overwrite=False,
    append=True,
    dataset_version=None,
    generate_if_missing=True,
    seed=None,
    note="Auto pipeline run"
):
    engine = get_engine()

    # --------------------------
    # 🔥 ALWAYS GENERATE NEW DATA
    # --------------------------
    print("📊 Generating new dataset...")

    meta = generate(
        dataset_version=dataset_version,   # ✅ FIXED
        seed=seed
    )

    # safety check
    if not meta or "file" not in meta:
        raise ValueError("Dataset generation failed — no file returned")

    file_path = meta["file"]
    dataset_ver = meta.get("dataset_version", "unknown")

    # --------------------------
    # Load CSV
    # --------------------------
    df = pd.read_csv(file_path)

    # --------------------------
    # Ensure dataset_version column
    # --------------------------
    if "dataset_version" not in df.columns:
        df["dataset_version"] = dataset_ver

    # --------------------------
    # Check existing rows
    # --------------------------
    with engine.connect() as conn:
        try:
            cnt = conn.execute("SELECT COUNT(1) FROM terms_enrollment").scalar()
        except Exception:
            cnt = 0

    print(f"📦 Existing rows in DB: {cnt}")

    # --------------------------
    # Load strategy
    # --------------------------
    if overwrite:
        print("⚠️ Overwriting table...")
        df.to_sql("terms_enrollment", engine, if_exists="replace", index=False)

    elif append:
        print("➕ Appending new data...")
        df.to_sql("terms_enrollment", engine, if_exists="append", index=False)

    else:
        if cnt == 0:
            print("📥 First load → inserting dataset...")
            df.to_sql("terms_enrollment", engine, if_exists="replace", index=False)
        else:
            print("⏭️ Data already exists — skipping load")

    # --------------------------
    # Track dataset version
    # --------------------------
    rows = len(df)

    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO dataset_versions 
        (dataset_version, created_at, rows_generated, notes)
        VALUES (?, datetime('now'), ?, ?)
        """,
        (dataset_ver, rows, note)
    )
    conn.commit()
    conn.close()

    print(f"✅ Dataset loaded | Version: {dataset_ver} | Rows: {rows}")

    return meta


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    print("\n🚀 LOADING DATA PIPELINE\n")

    create_tables()

    load_dataset(
        append=True,
        seed=None,
        note="Run via main.py"
    )

    print("\n✅ DATA PIPELINE COMPLETE\n")