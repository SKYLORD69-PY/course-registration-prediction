import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import sqlite3
from sqlalchemy import create_engine

from project_config import DATABASE_URL, DATABASE_FILE


def get_engine():
    """Return SQLAlchemy engine"""
    engine = create_engine(DATABASE_URL)
    return engine


def get_connection():
    """Return SQLite connection"""
    conn = sqlite3.connect(DATABASE_FILE)
    return conn


def test_connection():
    """Test database connection"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT sqlite_version();")
    version = cursor.fetchone()

    print("Connected to SQLite successfully")
    print("SQLite version:", version)

    conn.close()


if __name__ == "__main__":
    test_connection()