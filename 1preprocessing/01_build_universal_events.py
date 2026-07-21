"""
01_build_universal_events.py

Builds universal_events.parquet for the FINE-TUNING pipeline: combines
Diagnoses (DX) + discretized Labs/Vitals, applying the same temporal
censoring (GOLDEN RULE) used in generate_pretrain_data.py -- for diabetic
patients, only events BEFORE their Index_Date (first diagnosis date) are
kept; controls keep their entire history.

IMPORTANT (confirmed design decision): medications (RX) are NOT included.
Features are DX + LABS + VITALS, matching the pretraining feature set.
Demographics (gender/race/ethnicity) are also not included as tokens -- they
are only used to compute AGE (per-event age), same as in pretraining.
"""

import duckdb

con = duckdb.connect()

print("Combining Diagnoses + Labs/Vitals and applying temporal censoring (data leakage prevention)...")

query = """
    WITH Diagnosis AS (
        SELECT
            PATIENT_ID,
            CAST(Shifted_date AS DATE) AS DATE,
            'DX:' || CAST(DX AS VARCHAR) AS CODE
        FROM read_csv_auto('0data/deid_visit_dx.csv')
    ),
    LabsVitals AS (
        SELECT
            PATIENT_ID,
            CAST(DATE AS DATE) AS DATE,
            CODE
        FROM read_csv_auto('0data/discretized_labs_vitals.csv')
    ),
    Universal_Events AS (
        SELECT * FROM Diagnosis
        UNION ALL
        SELECT * FROM LabsVitals
    ),
    Diabetes_dates AS (
        SELECT
            PATIENT_ID,
            CAST(EARLIEST_DX AS DATE) AS Index_Date
        FROM read_csv_auto('0data/EARLIEST_dx_deid.csv')
    )
    -- GOLDEN RULE:
    -- If the patient is not in the diabetes table (Index_Date IS NULL),
    -- include their entire history (they are a control).
    -- If they are diabetic, only include events BEFORE their diagnosis (<).
    SELECT
        u.PATIENT_ID,
        u.DATE,
        u.CODE
    FROM Universal_Events u
    LEFT JOIN Diabetes_dates f ON u.PATIENT_ID = f.PATIENT_ID
    WHERE f.Index_Date IS NULL OR u.DATE < f.Index_Date
"""

con.execute(f"COPY ({query}) TO '0data/universal_events.parquet' (FORMAT PARQUET);")

print("Success! Censored events (DX + LABS + VITALS) saved to '0data/universal_events.parquet'.")

# Quick sanity-check summary
summary = con.execute("""
    SELECT
        CASE WHEN CODE LIKE 'DX:%' THEN 'DX' ELSE 'LAB/VITAL' END AS type,
        COUNT(*) AS n
    FROM read_parquet('0data/universal_events.parquet')
    GROUP BY 1
""").df()
print("\nCensored events summary:")
print(summary.to_string(index=False))