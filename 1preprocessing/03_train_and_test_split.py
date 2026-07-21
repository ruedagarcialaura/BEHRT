"""
03_train_and_test_split.py

Splits behrt_finetuning_data.parquet into train/test sets with labels.

IMPORTANT: the sequence data ('code'/'age' columns) is kept entirely in
DuckDB for both reading AND writing. Converting it to a pandas DataFrame and
then calling df.to_parquet() goes through pyarrow.Table.from_pandas(), which
is a DIFFERENT (and much less robust) code path than DuckDB's native
read/write for huge list columns -- this is what was crashing with
ArrowMemoryError on df_train.to_parquet(), even though DuckDB itself handles
these same huge lists (patients with 200k+ tokens) without any issue.

Only a small (patid, label) table -- no list columns -- is ever converted to
pandas, because sklearn's train_test_split needs it for the stratified
split. The actual sequence rows are filtered and written by DuckDB using the
patient ID sets computed from that split.
"""

import duckdb
from sklearn.model_selection import train_test_split

con = duckdb.connect()

print("1. Extracting AIM_GROUP label from the database...")
label_query = """
    SELECT DISTINCT PATIENT_ID, AIM_GROUP
    FROM read_csv_auto('0data/deid_visit_dx.csv')
"""
con.execute(f"CREATE OR REPLACE TEMP TABLE raw_labels AS ({label_query})")

print("2. Building a small (patid, label, code_len) table for the split "
      "-- no sequence data is materialized here...")
combined_query = """
    SELECT
        seq.patid,
        CASE lbl.AIM_GROUP
            WHEN '1_NoD' THEN 0
            WHEN '2_Type2' THEN 1
            ELSE NULL
        END AS label,
        len(seq.code) AS code_len
    FROM read_parquet('0data/behrt_finetuning_data.parquet') seq
    INNER JOIN raw_labels lbl ON seq.patid = lbl.PATIENT_ID
"""
con.execute(f"CREATE OR REPLACE TEMP TABLE patient_labels AS ({combined_query})")

# Filter: valid label + at least 2 codes of history (same rule as before)
con.execute("""
    CREATE OR REPLACE TEMP TABLE patient_labels_filtered AS
    SELECT patid, label
    FROM patient_labels
    WHERE label IS NOT NULL AND code_len >= 2
""")

patient_labels_df = con.execute("SELECT * FROM patient_labels_filtered").df()
print(f"   {len(patient_labels_df)} patients with a valid label and >= 2 codes.")

print("3. Splitting into Train (80%) and Test (20%) -- stratified, "
      "computed only on the small patid/label table...")
train_ids_df, test_ids_df = train_test_split(
    patient_labels_df,
    test_size=0.2,
    random_state=42,
    stratify=patient_labels_df['label']
)

# Register the small id sets back into DuckDB so we can filter the big
# sequence table with a plain SQL JOIN/IN -- still no pandas conversion of
# the sequence data itself.
con.register('train_ids', train_ids_df)
con.register('test_ids', test_ids_df)

print("4. Writing train/test sequence parquet files (entirely in DuckDB)...")

for split_name, ids_table in [('train', 'train_ids'), ('test', 'test_ids')]:
    out_query = f"""
        SELECT seq.patid, seq.code, seq.age, ids.label
        FROM read_parquet('0data/behrt_finetuning_data.parquet') seq
        INNER JOIN {ids_table} ids ON seq.patid = ids.patid
    """
    con.execute(f"COPY ({out_query}) TO '0data/diabetes_{split_name}.parquet' (FORMAT PARQUET);")
    print(f"   Saved 0data/diabetes_{split_name}.parquet")

print("--------------------------------------------------")
print("DATA PREPARATION COMPLETED!")
print(f"Total patients for Training (Train): {len(train_ids_df)}")
print(f"Total patients for Testing (Test): {len(test_ids_df)}")
print("\nLabel distribution in Train:")
print(train_ids_df['label'].value_counts())
print("--------------------------------------------------")
print("Next stop: Google Colab for Fine-Tuning!")