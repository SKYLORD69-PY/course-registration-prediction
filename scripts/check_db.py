# scripts/check_db.py
from database.db_connection import get_engine
import pandas as pd
engine = get_engine()
df = pd.read_sql("SELECT dataset_version, COUNT(*) as cnt FROM terms_enrollment GROUP BY dataset_version", engine)
print(df)
print(pd.read_sql("SELECT school, term_label, COUNT(*) as cnt FROM terms_enrollment GROUP BY school, term_label", engine).head(20))