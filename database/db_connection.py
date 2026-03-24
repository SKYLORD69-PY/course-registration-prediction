import os
import sys
import sqlite3
from sqlalchemy import create_engine

# -----------------------------
# Add project root
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from project_config import DATABASE_URL, DATABASE_FILE


# -----------------------------
# SQLAlchemy Engine
# -----------------------------
def get_engine():
    """Return SQLAlchemy engine (optimized for SQLite)"""

    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True
    )

    return engine


# -----------------------------
# SQLite Connection
# -----------------------------
def get_connection():
    """Return SQLite connection with timeout"""

    conn = sqlite3.connect(
        DATABASE_FILE,
        timeout=30  # 🔥 prevents 'database is locked'
    )

    return conn


# -----------------------------
# Test DB Connection
# -----------------------------
def test_connection():

    print(f"\n📦 Database Path: {DATABASE_FILE}\n")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT sqlite_version();")
    version = cursor.fetchone()

    print("✅ Connected to SQLite successfully")
    print("SQLite version:", version)

    conn.close()


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    test_connection()