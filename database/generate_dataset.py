# synth_generator_final_v2.py
"""
Final generator v2.

Key changes from earlier versions:
- Removed num_compulsory_eligible column.
- Added warmup period (warmup_years) to populate prev_term_enrollment and prev2_term_enrollment.
- Correctly add new Year1 intake to compulsory pool only at 'independence' term of each academic year.
- Separate school population vs enrolled students.
- Output column 'enrollment_this_term' (you can use this as target shifted by 1 term for next-term prediction).
"""

import random
import numpy as np
import pandas as pd
from datetime import date

random.seed(42)
np.random.seed(42)

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
    ("festivals",   10, 0.90),
    ("republic",    1, 0.80),
    ("colors",      4, 0.65),
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
    if term_label == "independence":
        return float(np.round(np.random.uniform(26.0, 28.0), 2))
    elif term_label == "festivals":
        return float(np.round(np.random.uniform(15.0, 17.0), 2))
    elif term_label == "republic":
        return float(np.round(np.random.uniform(4.0, 6.0), 2))
    elif term_label == "colors":
        return float(np.round(np.random.uniform(0.0, 3.0), 2))
    else:
        return float(np.round(np.random.uniform(0.0, 10.0), 2))

def _make_term_sequence(start_year, years, warmup_years):
    # returns list of (year, term_label, month, term_multiplier) covering warmup + requested years
    seq = []
    start = start_year - warmup_years
    end = start_year + years - 1
    for y in range(start, end + 1):
        for term_label, month, mult in TERMS:
            seq.append((y, term_label, month, mult))
    return seq

# -------------------------
# Generator
# -------------------------
def generate(years=6, start_year=2020, warmup_years=2, out_csv="engineering_workshop_term_school_final_v2.csv"):
    """
    years: number of *output* years (not counting warmup)
    warmup_years: how many years BEFORE start_year to simulate as warmup (default 2)
    """
    rows_all = []

    # initial intake (Year1) per school
    intake_by_school = {s: random.randint(info["min_intake"], info["max_intake"]) for s, info in SCHOOLS.items()}

    # prev tracker: per school [prev_term_enrollment, prev2_term_enrollment]
    prev_tracker = {s: [0, 0] for s in SCHOOLS.keys()}

    # compulsory pool per school (number of students that still need to do the workshop sometime)
    compulsory_pool = {s: None for s in SCHOOLS.keys()}

    # Build chronological sequence (including warmup)
    seq = _make_term_sequence(start_year, years, warmup_years)

    for (year_iter, term_label, month, term_multiplier) in seq:
        term_year = year_iter if month >= 7 else year_iter + 1
        term_start = date(term_year, month, 1)

        # If this is the 'independence' term for the academic year, apply growth to intake (new admits arrive)
        is_independence = (term_label == "independence")
        if is_independence:
            # apply year-on-year growth to intake
            growth = 1.0 + random.uniform(0.02, 0.07)
            for s in intake_by_school:
                intake_by_school[s] = max(1, int(round(intake_by_school[s] * growth)))

        for school in SCHOOLS.keys():
            # ---------- population snapshot (school population by year)
            # We use the current intake as year1 population, then small jitter for other years
            school_year1_population = intake_by_school[school]
            school_year2_population = int(max(0, round(intake_by_school[school] * random.uniform(0.88, 1.06))))
            school_year3_population = int(max(0, round(intake_by_school[school] * random.uniform(0.78, 0.96))))
            school_year4_population = int(max(0, round(intake_by_school[school] * random.uniform(0.68, 0.92))))
            total_students = (school_year1_population + school_year2_population +
                              school_year3_population + school_year4_population)

            # initialize compulsory pool first time we know total_students for the school
            if compulsory_pool[school] is None:
                # eligible pool equals total_students for VSST else 0
                compulsory_pool[school] = total_students if school == "VSST" else 0

            # If new intake arrived at this independence term (is_independence) then add those new Year1 students
            # to VSST compulsory pool (they become eligible). For non-VSST, no compulsory effect.
            if is_independence and school == "VSST":
                # Add the new cohort that just arrived
                compulsory_pool[school] += school_year1_population

            # ---------- credits
            avg_remaining_credits = jitter_remaining_credits(term_label)

            # ---------- base expected (non-compulsory) demand
            base_rate = random.uniform(BASE_RATE_LOW, BASE_RATE_HIGH)
            elective_rate = random.uniform(ELECTIVE_RATE_LOW, ELECTIVE_RATE_HIGH)
            base_interest = total_students * base_rate
            elective_component = total_students * elective_rate

            if school == "VSST":
                ge_component = 0.0
            else:
                ge_strength = (avg_remaining_credits / 40.0)
                ge_component = total_students * GE_BASE_RATE * ge_strength

            raw_expected = (base_interest + elective_component + ge_component) * term_multiplier

            # ---------- momentum
            prev_term = prev_tracker[school][0]
            prev2_term = prev_tracker[school][1]
            raw_expected += 0.10 * prev_term + 0.05 * prev2_term

            # ---------- compulsory taken this term (from pool)
            if compulsory_pool[school] > 0:
                drain_frac = random.uniform(COMPULSORY_DRAIN_MIN, COMPULSORY_DRAIN_MAX)
                candidate = int(round(compulsory_pool[school] * drain_frac * term_multiplier))
                num_compulsory = min(compulsory_pool[school], candidate, total_students)
            else:
                num_compulsory = 0

            compulsory_pool[school] = max(0, int(round(compulsory_pool[school] - num_compulsory)))

            # ---------- combine and noise
            noise = np.random.normal(0, NOISE_STD)
            enrollment_est = int(round(max(0, raw_expected + num_compulsory + noise)))

            if enrollment_est < num_compulsory:
                enrollment_est = int(num_compulsory)

            # ---------- allocate non-compulsory reasons
            non_comp = enrollment_est - num_compulsory
            if school == "VSST":
                # VSST: ge = 0
                if non_comp > 0:
                    num_elective = int(round(non_comp * random.uniform(0.55, 0.85)))
                    num_other = non_comp - num_elective
                else:
                    num_elective = num_other = 0
                num_ge = 0
            else:
                if non_comp > 0:
                    ge_frac = random.uniform(0.45, 0.70)
                    num_ge = int(round(non_comp * ge_frac))
                    rem_after_ge = non_comp - num_ge
                    num_elective = int(round(rem_after_ge * random.uniform(0.55, 0.85)))
                    num_other = rem_after_ge - num_elective
                else:
                    num_ge = num_elective = num_other = 0

            # ---------- final target: enrollment_this_term
            enrollment_this_term = num_compulsory + num_ge + num_elective + num_other

            # ---------- distribute enrolled students across study years
            if enrollment_this_term > 0:
                enrolled_year1, enrolled_year2, enrolled_year3, enrolled_year4 = \
                    np.random.multinomial(enrollment_this_term, YEAR_WEIGHTS)
            else:
                enrolled_year1 = enrolled_year2 = enrolled_year3 = enrolled_year4 = 0

            # ---------- recent trend (before updating prevs)
            recent_trend = prev_term - prev2_term

            # ---------- update prev tracker
            prev_tracker[school][1] = prev_tracker[school][0]
            prev_tracker[school][0] = enrollment_this_term

            # ---------- append row (we include prevs so warmup rows will have prevs for later)
            rows_all.append({
                "term_start_date": term_start,
                "year": year_iter,
                "term_label": term_label,
                "school": school,
                # population
                "school_year1_population": int(school_year1_population),
                "school_year2_population": int(school_year2_population),
                "school_year3_population": int(school_year3_population),
                "school_year4_population": int(school_year4_population),
                "total_students_in_school": int(total_students),
                # enrollment (workshop) by study year
                "enrolled_year1": int(enrolled_year1),
                "enrolled_year2": int(enrolled_year2),
                "enrolled_year3": int(enrolled_year3),
                "enrolled_year4": int(enrolled_year4),
                "enrollment_this_term": int(enrollment_this_term),
                # reasons
                "num_compulsory": int(num_compulsory),
                "num_ge": int(num_ge),
                "num_elective": int(num_elective),
                "num_other": int(num_other),
                # credits & lag
                "avg_remaining_credits": float(avg_remaining_credits),
                "prev_term_enrollment": int(prev_term),
                "prev2_term_enrollment": int(prev2_term),
                "recent_trend": int(recent_trend),
            })

    # -------------------------
    # Build DataFrame and filter out warmup
    # -------------------------
    df_all = pd.DataFrame(rows_all)

    # warmup ends at start_year - 1? We built seq starting (start_year - warmup_years).
    # Keep only rows whose year >= start_year
    df_out = df_all[df_all["year"] >= start_year].reset_index(drop=True)

    # -------------------------
    # Post-fixes: enforce invariants
    # -------------------------
    # total_students_in_school == sum(pop columns)
    pop_sum = df_out[["school_year1_population", "school_year2_population",
                      "school_year3_population", "school_year4_population"]].sum(axis=1)
    df_out["total_students_in_school"] = pop_sum.astype(int)

    # enrollment_this_term == sum(enrolled_year*)
    enrolled_sum = df_out[["enrolled_year1", "enrolled_year2", "enrolled_year3", "enrolled_year4"]].sum(axis=1)
    mismatch = df_out["enrollment_this_term"] - enrolled_sum
    if mismatch.abs().sum() != 0:
        df_out["enrolled_year4"] = df_out["enrolled_year4"] + mismatch

    # enrollment_this_term == sum(reason columns
    reason_sum = df_out[["num_compulsory", "num_ge", "num_elective", "num_other"]].sum(axis=1)
    mismatch2 = df_out["enrollment_this_term"] - reason_sum
    if mismatch2.abs().sum() != 0:
        df_out["num_other"] = df_out["num_other"] + mismatch2

    # final asserts (should hold)
    assert (df_out["total_students_in_school"] ==
            df_out[["school_year1_population", "school_year2_population",
                    "school_year3_population", "school_year4_population"]].sum(axis=1)).all(), "Population invariant failed"
    assert (df_out["enrollment_this_term"] ==
            df_out[["enrolled_year1", "enrolled_year2", "enrolled_year3", "enrolled_year4"]].sum(axis=1)).all(), "Enrollment-year invariant failed"
    assert (df_out["enrollment_this_term"] ==
            df_out[["num_compulsory", "num_ge", "num_elective", "num_other"]].sum(axis=1)).all(), "Reason invariant failed"

    # Save
    df_out.to_csv(out_csv, index=False)
    print(f"Generated {out_csv} with {len(df_out)} rows (warmup_years={warmup_years})")
    return df_out

# -------------------------
# If run as script
# -------------------------
if __name__ == "__main__":
    # default: produce 6 years starting 2020 with 2-year warmup (so prevs populated)
    df = generate(years=6, start_year=2020, warmup_years=2)
    # show a few rows to inspect
    with pd.option_context("display.max_rows", 12, "display.max_columns", None):
        print(df.head(12).to_string(index=False))