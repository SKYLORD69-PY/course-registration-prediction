import os
import sys
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from database.db_connection import get_engine, get_connection
from database.generate_dataset import generate
from project_config import DATASET_FILE


def create_tables():
    """Execute SQL schema to create database tables"""

    print("Creating database tables...")

    conn = get_connection()
    cursor = conn.cursor()

    sql_file = os.path.join(PROJECT_ROOT, "database", "create_tables.sql")

    with open(sql_file, "r") as f:
        sql_script = f.read()

    cursor.executescript(sql_script)

    conn.commit()
    conn.close()

    print("Tables created successfully")


def load_dataset():
    """Generate dataset if needed and load it into SQLite"""

    # Ensure data directory exists
    os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)

    # Generate dataset if it doesn't exist
    if not os.path.exists(DATASET_FILE):
        print("Dataset not found. Generating dataset...")
        generate(out_csv=DATASET_FILE)

    print("Reading dataset...")

    df = pd.read_csv(DATASET_FILE)

    engine = get_engine()

    print("Inserting data into terms_enrollment table...")

    df.to_sql(
        "terms_enrollment",
        engine,
        if_exists="append",
        index=False
    )

    print("Dataset successfully loaded into SQLite!")


if __name__ == "__main__":

    create_tables()

    load_dataset()