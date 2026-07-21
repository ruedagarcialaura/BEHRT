import duckdb
import os

os.makedirs("1preprocessing/preprocessed_data", exist_ok=True)

print("==================================================")
print(" PHASE 1: EXTRACTING CENSORED HISTORY (LEAKAGE-SAFE) ")
print("==================================================")

# 1. Connect to DuckDB
con = duckdb.connect()

# Combine Diagnoses and Labs/Vitals (NO MEDICATIONS).
# IMPORTANT: apply the SAME temporal censoring rule as 01_build_universal_events.py.
# For diabetic patients, only events strictly BEFORE their Index_Date (first T2D
# diagnosis) are kept. This prevents the MLM pretraining from ever seeing the
# diabetes diagnosis code itself, or any post-diagnosis labs/vitals/complications,
# for the exact patients that later become the fine-tuning cohort.
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
    Uncensored_Events AS (
        SELECT * FROM Diagnosis
        UNION ALL
        SELECT * FROM LabsVitals
    ),
    Diabetes_dates AS (
        -- CRITICAL FIX: EARLIEST_dx_deid.csv has ONE ROW PER (PATIENT_ID, DX)
        -- pair, not one row per patient -- each distinct diagnosis code has
        -- its own "first occurrence" date. Without MIN()+GROUP BY, the JOIN
        -- below would match each event against ALL of a patient's index
        -- dates, and an event only needs to be before ONE of them to pass
        -- the WHERE filter -- letting events AFTER the patient's TRUE
        -- earliest diagnosis leak through as long as they're before some
        -- LATER "first occurrence of a different diabetes code" date.
        SELECT
            PATIENT_ID,
            MIN(CAST(EARLIEST_DX AS DATE)) AS Index_Date
        FROM read_csv_auto('0data/EARLIEST_dx_deid.csv')
        GROUP BY PATIENT_ID
    )
    -- GOLDEN RULE (same as 01_build_universal_events.py):
    -- Non-diabetic patients (no Index_Date) keep their full history.
    -- Diabetic patients only keep events strictly before their Index_Date.
    SELECT 
        u.PATIENT_ID, 
        u.DATE, 
        u.CODE
    FROM Uncensored_Events u
    LEFT JOIN Diabetes_dates f ON u.PATIENT_ID = f.PATIENT_ID
    WHERE f.Index_Date IS NULL OR u.DATE < f.Index_Date
"""

print("Executing DuckDB query to combine and censor all events...")
con.execute(f"COPY ({query}) TO '1preprocessing/preprocessed_data/temp_pretrain_events_censored.parquet' (FORMAT PARQUET);")


print("\n==================================================")
print(" PHASE 1.5: CALCULATING AGE PER EVENT (in DuckDB) ")
print("==================================================")
# IMPORTANT: this used to be a pandas `events_df.merge(patient_info, ...)`
# on the full ~80M-row events table, which is exactly the kind of operation
# that blows up with ArrowMemoryError on a PyArrow-backed pandas frame this
# large. Doing the join/aggregation in DuckDB instead avoids ever
# materializing that merge as a pandas object -- only the small, already
# age-annotated result gets written out.
age_query = """
    WITH Events AS (
        SELECT PATIENT_ID, DATE, CODE
        FROM read_parquet('1preprocessing/preprocessed_data/temp_pretrain_events_censored.parquet')
    ),
    LastVisit AS (
        SELECT PATIENT_ID, MAX(DATE) AS LAST_VISIT_DATE
        FROM Events
        GROUP BY PATIENT_ID
    ),
    Demo AS (
        SELECT PATIENT_ID, AGE_AT_END
        FROM read_csv_auto('0data/deid_DEM.csv')
    ),
    PatientInfo AS (
        SELECT
            lv.PATIENT_ID,
            EXTRACT(YEAR FROM lv.LAST_VISIT_DATE) - d.AGE_AT_END AS YOB
        FROM LastVisit lv
        INNER JOIN Demo d ON lv.PATIENT_ID = d.PATIENT_ID
    )
    SELECT
        e.PATIENT_ID,
        e.DATE,
        e.CODE,
        CAST(GREATEST(EXTRACT(YEAR FROM e.DATE) - p.YOB, 0) AS VARCHAR) AS AGE
    FROM Events e
    INNER JOIN PatientInfo p ON e.PATIENT_ID = p.PATIENT_ID
    ORDER BY e.PATIENT_ID, e.DATE
"""
print("Executing DuckDB query to join demographics and compute AGE per event...")
con.execute(
    f"COPY ({age_query}) TO "
    f"'1preprocessing/preprocessed_data/temp_pretrain_events_with_age.parquet' (FORMAT PARQUET);"
)


print("\n==================================================")
print(" PHASE 2: BUILDING LONGITUDINAL SEQUENCES (in DuckDB) ")
print("==================================================")
# IMPORTANT: this used to build sequences with pandas chunked groupby-apply
# and then call pandas' df.to_parquet(), which goes through
# pyarrow.Table.from_pandas() -- a DIFFERENT and much less robust code path
# for huge list columns than DuckDB's native read/write. That is what was
# crashing with ArrowMemoryError even though DuckDB itself handles these
# same huge lists without any issue (as already proven in
# 02_format_for_finetuning.py). Rewritten here to keep everything in DuckDB,
# split into PATIENT BUCKETS since DuckDB's list()/array_agg() aggregation
# does not support spilling to disk -- bucketing bounds how many patients'
# worth of data any single query has to hold in memory at once.

TMP_DIR = "1preprocessing/preprocessed_data"
N_BUCKETS = 20  # increase if you still hit OutOfMemory with your real data

os.makedirs(f"{TMP_DIR}/pretrain_output_buckets", exist_ok=True)

# Stage 2a: assign each patient to a bucket, partitioned on disk
print("\n[1/3] Assigning patient buckets...")
bucket_query = f"""
    SELECT *, ABS(HASH(PATIENT_ID)) % {N_BUCKETS} AS bucket
    FROM read_parquet('{TMP_DIR}/temp_pretrain_events_with_age.parquet')
"""
con.execute(
    f"COPY ({bucket_query}) TO '{TMP_DIR}/pretrain_events_bucketed' "
    f"(FORMAT PARQUET, PARTITION_BY (bucket));"
)
print("Bucket assignment done.")

# Stage 2b: process one bucket of patients at a time -- group into visits
# AND flatten into per-patient sequences in a single query per bucket.
# NOTE: events on the same calendar day have no meaningful sub-day ordering
# (truncated to DATE granularity), so 'ORDER BY CODE' just makes the
# within-day order deterministic/reproducible instead of depending on
# arbitrary scan order.
print(f"\n[2/3] Processing {N_BUCKETS} patient buckets...")
for b in range(N_BUCKETS):
    bucket_dir = f"{TMP_DIR}/pretrain_events_bucketed/bucket={b}"
    if not os.path.exists(bucket_dir):
        print(f"  Bucket {b}: no data, skipping.")
        continue

    output_path = f"{TMP_DIR}/pretrain_output_buckets/bucket_{b}.parquet"
    if os.path.exists(output_path):
        print(f"  Bucket {b}: already processed, skipping.")
        continue

    bucket_query = f"""
        WITH VisitLevel AS (
            SELECT
                PATIENT_ID,
                DATE,
                list(CODE ORDER BY CODE) || ['SEP'] AS codes_seq,
                list(AGE ORDER BY CODE) || [first(AGE)] AS ages_seq
            FROM read_parquet('{bucket_dir}/*.parquet')
            GROUP BY PATIENT_ID, DATE
        ),
        PatientLevel AS (
            SELECT
                PATIENT_ID AS patid,
                flatten(list(codes_seq ORDER BY DATE)) AS code,
                flatten(list(ages_seq ORDER BY DATE)) AS age
            FROM VisitLevel
            GROUP BY PATIENT_ID
        )
        -- Filter out patients with less than 2 codes (the model needs
        -- context to learn).
        SELECT * FROM PatientLevel WHERE len(code) >= 2
    """
    con.execute(f"COPY ({bucket_query}) TO '{output_path}' (FORMAT PARQUET);")
    print(f"  Bucket {b}: done.")

print("Stage 2 done.")

# Stage 2c: concatenate all per-bucket outputs into the final parquet
print("\n[3/3] Concatenating all buckets into the final file...")
concat_query = f"SELECT * FROM read_parquet('{TMP_DIR}/pretrain_output_buckets/bucket_*.parquet')"
con.execute(
    f"COPY ({concat_query}) TO "
    f"'{TMP_DIR}/behrt_pretrain_data.parquet' (FORMAT PARQUET);"
)

print("\n==================================================")
print(" SUCCESS! PRE-TRAINING DATA READY ")
print(f" Saved as: '{TMP_DIR}/behrt_pretrain_data.parquet' ")
print("==================================================")

summary = con.execute(f"""
    SELECT
        COUNT(*) AS n_patients,
        AVG(len(code)) AS avg_seq_len,
        MAX(len(code)) AS max_seq_len
    FROM read_parquet('{TMP_DIR}/behrt_pretrain_data.parquet')
""").df()
print("\nSummary:")
print(summary.to_string(index=False))