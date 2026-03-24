-- ==============================
-- Main dataset table
-- ==============================

CREATE TABLE IF NOT EXISTS terms_enrollment (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    term_start_date TEXT,
    year INTEGER,
    term_label TEXT,
    school TEXT,

    school_year1_population INTEGER,
    school_year2_population INTEGER,
    school_year3_population INTEGER,
    school_year4_population INTEGER,
    total_students_in_school INTEGER,

    enrolled_year1 INTEGER,
    enrolled_year2 INTEGER,
    enrolled_year3 INTEGER,
    enrolled_year4 INTEGER,

    enrollment_this_term INTEGER,

    num_compulsory INTEGER,
    num_ge INTEGER,
    num_elective INTEGER,
    num_other INTEGER,

    avg_remaining_credits REAL,
    prev_term_enrollment INTEGER,
    prev2_term_enrollment INTEGER,
    recent_trend INTEGER,

    dataset_version TEXT
);

-- 🔥 Index for fast filtering
CREATE INDEX IF NOT EXISTS idx_dataset_version
ON terms_enrollment(dataset_version);


-- ==============================
-- Model history
-- ==============================

CREATE TABLE IF NOT EXISTS model_history (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    model_version TEXT,
    model_type TEXT,
    training_date TEXT,

    mae REAL,
    rmse REAL,
    r2 REAL,
    mape REAL,

    training_rows INTEGER,   

    model_path TEXT
);


-- ==============================
-- Dataset version tracking
-- ==============================

CREATE TABLE IF NOT EXISTS dataset_versions (

    dataset_version TEXT PRIMARY KEY,
    created_at TEXT,
    rows_generated INTEGER,
    notes TEXT
);