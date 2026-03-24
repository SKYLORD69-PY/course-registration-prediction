# database/generate_dataset.py

import os
import random
import numpy as np
import pandas as pd
from datetime import date
import time


# -------------------------
# Config
# -------------------------
SCHOOLS = {
    "VSST": {"min_intake": 50, "max_intake": 60},
    "TSM": {"min_intake": 25, "max_intake": 30},
    "JAGSoM": {"min_intake": 25, "max_intake": 30},
    "VSOD": {"min_intake": 5, "max_intake": 10},
    "VSOL": {"min_intake": 5, "max_intake": 10},
}

TERMS = [
    ("independence", 7, 1.00),
    ("festivals", 10, 0.90),
    ("republic", 1, 0.80),
    ("colors", 4, 0.65),
]

YEAR_WEIGHTS = [0.45, 0.35, 0.15, 0.05]

BASE_RATE_LOW, BASE_RATE_HIGH = 0.06, 0.12
ELECTIVE_RATE_LOW, ELECTIVE_RATE_HIGH = 0.015, 0.03
GE_BASE_RATE = 0.08

COMPULSORY_DRAIN_MIN, COMPULSORY_DRAIN_MAX = 0.15, 0.35
NOISE_STD = 4.0


# -------------------------
# Helpers
# -------------------------
def jitter_remaining_credits(term_label):
    ranges = {
        "independence": (26.0, 28.0),
        "festivals": (15.0, 17.0),
        "republic": (4.0, 6.0),
        "colors": (0.0, 3.0),
    }
    low, high = ranges.get(term_label, (0.0, 10.0))
    return float(np.round(np.random.uniform(low, high), 2))


def _make_term_sequence(start_year, years, warmup_years):
    seq = []
    start = start_year - warmup_years
    end = start_year + years - 1

    for y in range(start, end + 1):
        for term_label, month, mult in TERMS:
            seq.append((y, term_label, month, mult))

    return seq


# -------------------------
# MAIN GENERATOR
# -------------------------
def generate(
    years=6,
    start_year=2020,
    warmup_years=2,
    out_csv=None,
    dataset_version=None,
    seed=None,
):

    # 🔥 AUTO VERSIONING
    if dataset_version is None:
        dataset_version = f"v{int(time.time())}"

    # 🔥 UNIQUE FILE PER RUN
    if out_csv is None:
        out_csv = f"data/dataset_{dataset_version}.csv"

    # 🔥 RANDOM SEED
    if seed is None:
        seed = int(time.time())

    random.seed(seed)
    np.random.seed(seed)

    rows_all = []

    intake_by_school = {
        s: random.randint(info["min_intake"], info["max_intake"])
        for s, info in SCHOOLS.items()
    }

    prev_tracker = {s: [0, 0] for s in SCHOOLS.keys()}
    compulsory_pool = {s: None for s in SCHOOLS.keys()}

    seq = _make_term_sequence(start_year, years, warmup_years)

    for (year_iter, term_label, month, term_multiplier) in seq:
        term_year = year_iter if month >= 7 else year_iter + 1
        term_start = date(term_year, month, 1)

        is_independence = term_label == "independence"

        if is_independence:
            growth = 1.0 + random.uniform(0.02, 0.07)
            for s in intake_by_school:
                intake_by_school[s] = max(
                    1, int(round(intake_by_school[s] * growth))
                )

        for school in SCHOOLS.keys():

            y1 = intake_by_school[school]
            y2 = int(y1 * random.uniform(0.88, 1.06))
            y3 = int(y1 * random.uniform(0.78, 0.96))
            y4 = int(y1 * random.uniform(0.68, 0.92))
            total_students = y1 + y2 + y3 + y4

            if compulsory_pool[school] is None:
                compulsory_pool[school] = total_students if school == "VSST" else 0

            if is_independence and school == "VSST":
                compulsory_pool[school] += y1

            avg_remaining_credits = jitter_remaining_credits(term_label)

            base_interest = total_students * random.uniform(BASE_RATE_LOW, BASE_RATE_HIGH)
            elective = total_students * random.uniform(ELECTIVE_RATE_LOW, ELECTIVE_RATE_HIGH)

            ge = 0
            if school != "VSST":
                ge = total_students * GE_BASE_RATE * (avg_remaining_credits / 40.0)

            raw_expected = (base_interest + elective + ge) * term_multiplier

            prev_term = prev_tracker[school][0]
            prev2_term = prev_tracker[school][1]

            raw_expected += 0.10 * prev_term + 0.05 * prev2_term

            if compulsory_pool[school] > 0:
                drain = random.uniform(COMPULSORY_DRAIN_MIN, COMPULSORY_DRAIN_MAX)
                num_compulsory = int(compulsory_pool[school] * drain * term_multiplier)
                num_compulsory = min(num_compulsory, compulsory_pool[school])
            else:
                num_compulsory = 0

            compulsory_pool[school] -= num_compulsory

            noise = np.random.normal(0, NOISE_STD)
            enrollment = int(max(0, raw_expected + num_compulsory + noise))

            if enrollment < num_compulsory:
                enrollment = num_compulsory

            non_comp = enrollment - num_compulsory

            if school == "VSST":
                num_elective = int(non_comp * 0.7)
                num_other = non_comp - num_elective
                num_ge = 0
            else:
                num_ge = int(non_comp * 0.6)
                rem = non_comp - num_ge
                num_elective = int(rem * 0.7)
                num_other = rem - num_elective

            enrollment_this_term = num_compulsory + num_ge + num_elective + num_other

            enrolled = np.random.multinomial(enrollment_this_term, YEAR_WEIGHTS)

            prev_tracker[school] = [enrollment_this_term, prev_term]

            rows_all.append({
                "term_start_date": term_start,
                "year": year_iter,
                "term_label": term_label,
                "school": school,
                "school_year1_population": y1,
                "school_year2_population": y2,
                "school_year3_population": y3,
                "school_year4_population": y4,
                "total_students_in_school": total_students,
                "enrolled_year1": enrolled[0],
                "enrolled_year2": enrolled[1],
                "enrolled_year3": enrolled[2],
                "enrolled_year4": enrolled[3],
                "enrollment_this_term": enrollment_this_term,
                "num_compulsory": num_compulsory,
                "num_ge": num_ge,
                "num_elective": num_elective,
                "num_other": num_other,
                "avg_remaining_credits": avg_remaining_credits,
                "prev_term_enrollment": prev_term,
                "prev2_term_enrollment": prev2_term,
                "recent_trend": prev_term - prev2_term,
                "dataset_version": dataset_version
            })

    df = pd.DataFrame(rows_all)
    df = df[df["year"] >= start_year].reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"✅ Generated {len(df)} rows | Version: {dataset_version}")

    return {
        "dataset_version": dataset_version,
        "rows": len(df),
        "file": out_csv
    }


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    meta = generate()
    print(meta)