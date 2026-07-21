"""
02_format_for_finetuning.py

Converts 0data/universal_events.parquet (DX + LABS + VITALS, already
censored by Index_Date) into the per-patient (code/age) sequences that
BEHRT consumes for fine-tuning.

Runs entirely in DuckDB, but with the visit-grouping/flattening step split
into PATIENT BUCKETS instead of one single query. This matters because
DuckDB's list()/array_agg() aggregation does not support spilling to disk
when it runs out of memory (unlike simple aggregates such as SUM/COUNT) --
even splitting the pipeline into sequential stages (age calculation, visit
grouping, flattening) is not enough if a single stage's list aggregation
still has to hold ALL ~200k patients' groups in memory at once. Bucketing
by patient bounds the number of patients processed by the list aggregation
in any single query, keeping peak memory manageable regardless of total
dataset size.

Stage 1: compute per-event AGE (SQL JOIN, cheap) and assign each patient to
         one of N_BUCKETS buckets via a hash of PATIENT_ID, writing the
         result partitioned by bucket.
Stage 2: for each bucket (one at a time), group into visits and flatten
         into per-patient sequences -- a small, bounded amount of data per
         query, regardless of how many patients/events the full dataset has.
Stage 3: concatenate all per-bucket outputs into the final parquet.
"""

import duckdb
import os
import glob

TMP_DIR = "0data/duckdb_tmp"
os.makedirs(TMP_DIR, exist_ok=True)

con = duckdb.connect()

# Adjust MEMORY_LIMIT_GB to roughly 70-80% of the RAM actually available in
# your environment. N_BUCKETS controls how many patients get processed by
# the list aggregation in a single query -- if you still hit OutOfMemory,
# increase N_BUCKETS (smaller buckets = less memory per query, more queries).
MEMORY_LIMIT_GB = 10
N_BUCKETS = 20

con.execute(f"SET memory_limit='{MEMORY_LIMIT_GB}GB';")
con.execute(f"PRAGMA temp_directory='{TMP_DIR}';")
con.execute("SET threads=4;")

print(f"DuckDB configured with memory_limit={MEMORY_LIMIT_GB}GB, "
      f"temp_directory='{TMP_DIR}', N_BUCKETS={N_BUCKETS}")

# ---------------------------------------------------------------------------
# Stage 1: per-event AGE + bucket assignment, partitioned by bucket on disk
# ---------------------------------------------------------------------------
print("\n[1/3] Computing per-event AGE and assigning patient buckets...")

stage1_query = f"""
    WITH Events AS (
        SELECT PATIENT_ID, DATE, CODE
        FROM read_parquet('0data/universal_events.parquet')
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
        CAST(GREATEST(EXTRACT(YEAR FROM e.DATE) - p.YOB, 0) AS VARCHAR) AS AGE,
        ABS(HASH(e.PATIENT_ID)) % {N_BUCKETS} AS bucket
    FROM Events e
    INNER JOIN PatientInfo p ON e.PATIENT_ID = p.PATIENT_ID
"""
con.execute(
    f"COPY ({stage1_query}) TO '{TMP_DIR}/events_with_age' "
    f"(FORMAT PARQUET, PARTITION_BY (bucket));"
)
print("Stage 1 done.")

# ---------------------------------------------------------------------------
# Stage 2: process one bucket of patients at a time -- group into visits
# AND flatten into per-patient sequences in a single query per bucket, since
# each bucket now only holds ~1/N_BUCKETS of the total patients.
# NOTE: events on the same calendar day have no meaningful sub-day ordering
# (we truncated to DATE granularity), so 'ORDER BY CODE' just makes the
# within-day order deterministic/reproducible instead of depending on
# arbitrary scan order.
# ---------------------------------------------------------------------------
print(f"\n[2/3] Processing {N_BUCKETS} patient buckets...")

os.makedirs(f"{TMP_DIR}/output_buckets", exist_ok=True)

for b in range(N_BUCKETS):
    bucket_dir = f"{TMP_DIR}/events_with_age/bucket={b}"
    if not os.path.exists(bucket_dir):
        print(f"  Bucket {b}: no data, skipping.")
        continue

    output_path = f"{TMP_DIR}/output_buckets/bucket_{b}.parquet"
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
        )
        SELECT
            PATIENT_ID AS patid,
            flatten(list(codes_seq ORDER BY DATE)) AS code,
            flatten(list(ages_seq ORDER BY DATE)) AS age
        FROM VisitLevel
        GROUP BY PATIENT_ID
    """
    con.execute(f"COPY ({bucket_query}) TO '{output_path}' (FORMAT PARQUET);")
    print(f"  Bucket {b}: done.")

print("Stage 2 done.")

# ---------------------------------------------------------------------------
# Stage 3: concatenate all per-bucket outputs into the final parquet
# ---------------------------------------------------------------------------
print("\n[3/3] Concatenating all buckets into the final file...")

concat_query = f"""
    SELECT * FROM read_parquet('{TMP_DIR}/output_buckets/bucket_*.parquet')
"""
con.execute(f"COPY ({concat_query}) TO '0data/behrt_finetuning_data.parquet' (FORMAT PARQUET);")

print("\nAbsolute success! Sequence data ready for the Train/Test split.")

# Quick sanity-check summary
summary = con.execute("""
    SELECT
        COUNT(*) AS n_patients,
        AVG(len(code)) AS avg_seq_len,
        MAX(len(code)) AS max_seq_len
    FROM read_parquet('0data/behrt_finetuning_data.parquet')
""").df()
print("\nSummary:")
print(summary.to_string(index=False))