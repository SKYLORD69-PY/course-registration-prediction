import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from database.db_connection import get_engine

engine = get_engine()

print("\nChecking database...\n")

# total rows
df = pd.read_sql("SELECT COUNT(*) as rows FROM terms_enrollment", engine)
print("Total rows in dataset:", df.iloc[0]["rows"])

print("\nRows per school:")

df = pd.read_sql(
"""
SELECT school, COUNT(*) as rows
FROM terms_enrollment
GROUP BY school
""",
engine
)

print(df)

print("\nRows per term:")

df = pd.read_sql(
"""
SELECT term_label, COUNT(*) as rows
FROM terms_enrollment
GROUP BY term_label
""",
engine
)

print(df)

print("\nSample rows:\n")

df = pd.read_sql(
"""
SELECT year, term_label, school, enrollment_this_term
FROM terms_enrollment
LIMIT 10
""",
engine
)

print(df)