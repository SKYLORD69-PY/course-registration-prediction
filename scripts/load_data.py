# scripts/load_data.py
import os, pandas as pd
from database.db_connection import get_engine, get_connection
from database.generate_dataset import generate
from project_config import DATASET_FILE, DATA_FOLDER

def create_tables():

    from database.db_connection import get_connection
    import os

    conn = get_connection()

    sql_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "database",
        "create_tables.sql"
    )

    with open(sql_path, "r") as f:
        sql_script = f.read()

    conn.executescript(sql_script)

    conn.commit()
    conn.close()

    print("Database tables created successfully")

def load_dataset(overwrite=False, append=False, dataset_version=None, generate_if_missing=True, seed=42, note=""):
    """
    overwrite   -> replace the terms_enrollment table
    append      -> append rows from CSV (must include dataset_version column)
    default     -> if table empty, load; else skip (safe)
    """
    engine = get_engine()

    # ensure CSV exists (generate if required)
    if not os.path.exists(DATASET_FILE):
        if generate_if_missing:
            meta = generate(out_csv=DATASET_FILE, dataset_version=dataset_version or f"v{pd.Timestamp.now().strftime('%Y%m%d%H%M')}", seed=seed)
        else:
            raise FileNotFoundError(DATASET_FILE)
    else:
        meta = None

    df = pd.read_csv(DATASET_FILE)
    if "dataset_version" not in df.columns:
        df["dataset_version"] = dataset_version or (meta["dataset_version"] if meta else "v_unknown")

    # check current row count
    with engine.connect() as conn:
        try:
            cnt = conn.execute("SELECT COUNT(1) FROM terms_enrollment").scalar()
        except Exception:
            cnt = 0

    if overwrite:
        df.to_sql("terms_enrollment", engine, if_exists="replace", index=False)
    elif append:
        df.to_sql("terms_enrollment", engine, if_exists="append", index=False)
    else:
        if cnt == 0:
            df.to_sql("terms_enrollment", engine, if_exists="replace", index=False)
        else:
            print(f"terms_enrollment already has {cnt} rows — skipping load (use overwrite=True or append=True).")

    # record dataset_versions row if meta present or dataset_version known
    dataset_ver = df["dataset_version"].iloc[0]
    if meta is None:
        rows = len(df)
        file_path = DATASET_FILE
    else:
        rows = meta["rows"]
        file_path = meta["file"]

    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO dataset_versions (dataset_version, created_at, rows_generated, notes) VALUES (?, datetime('now'), ?, ?)",
        (dataset_ver, rows, note)
    )
    conn.commit()
    conn.close()