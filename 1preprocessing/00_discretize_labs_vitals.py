"""
00_discretize_labs_vitals.py

Convierte labs y vitales en tokens discretos ("LAB:<test>_BIN_n") listos para
BEHRT. Reescrito para usar DuckDB en el procesamiento pesado en lugar de
pandas: DuckDB procesa los CSV de forma streaming/columnar sin necesitar
tener el fichero entero materializado en RAM como objetos Python, que es lo
que estaba provocando los ArrowMemoryError con pandas (tanto en la lectura
como luego en el dropna).

La única parte que sigue en pandas es la construcción, en Python, del SQL de
binning y del mapa de sinónimos -- son operaciones sobre diccionarios
pequeños (34 tests, 55 filas de dpi_lab_map.csv), no sobre los CSV grandes.
"""

import duckdb
import pandas as pd
import numpy as np

print("Starting Binning process for Labs and Vitals (DuckDB backend)...")

BINS_THRESHOLDS = {
    'ALT(SGPT)': [10, 49, 98, 245, 735],
    'AST(SGOT)': [10, 34, 68, 170, 510],
    'BUN': [6, 20, 50, 100, 250],
    'CHOLESTEROL': [200, 240],
    'HDL': [45, 80, 100],
    'HGB A1C': [5.7, 6.5],
    'POTASSIUM': [3.5, 5.1, 6.5],
    'TRIGLYCERIDE': [150, 200, 500],
    'VITAL_10541041_MEAN': [18.5, 25, 30, 35, 40],
    'VITAL_10541455_MEAN': [36.1, 37.3, 38.1, 39.1, 41.1],
    'VITAL_10541467_MEAN': [60, 100],
    'VITAL_10541503_MEAN': [12, 20],
    'VITAL_266705352_MEAN': [90, 92, 94, 97, 100],
    'VITAL_283305634_MEAN': [97.0, 99.1, 100.6, 102.4, 105.8],
    'VITAL_68924855_MEAN': [120, 130, 140, 180],
    'LDL-POCT': [100, 130, 160, 190],
    'VITAL_10541029_MEAN': [120, 130, 140, 181],
    'VITAL_34506073_MEAN': [12, 21, 31],
    'VITAL_68924858_MEAN': [89, 90, 95],
    'ALB CONC': [3.5, 5.1],
    'FERRITIN': [20, 250],
    'INR': [0.8, 1.2, 2.0, 3.1],
    'PROTHROMBIN TIME': [11.0, 13.6],
    'UR CREATININE': [20, 321],
    'UR TOTAL PROTEIN': [150, 3000],
    'm_Bilirubin.direct': [0.3],
    'VITAL_10155324_MEAN': [1, 4, 7],
    'VITAL_10155611_MEAN': [1, 24, 32, 46],
    'VITAL_10155613_MEAN': [22, 36, 61],
    'VITAL_10541434_MEAN': [32, 51],
    'VITAL_10541511_MEAN': [60, 101, 131],
    'VITAL_10541524_MEAN': [36.1, 37.3, 38.0],
    'VITAL_10541596_MEAN': [9, 13],
    'VITAL_14049161_MEAN': [40, 50, 100],
}


# ---------------------------------------------------------------------------
# 0. Synonym map from the SUNQUEST -> LOINC crosswalk (same logic as before,
#    dpi_lab_map.csv is tiny so this stays in pandas without any risk).
# ---------------------------------------------------------------------------
def build_lab_synonym_map(crosswalk_path, canonical_names):
    dpi = pd.read_csv(crosswalk_path)
    dpi['SUNQUEST_NAME'] = dpi['SUNQUEST_NAME'].astype(str).str.strip()

    name_to_loincs = dpi.groupby('SUNQUEST_NAME')['LOINC_CODE'].apply(set).to_dict()
    loinc_to_names = dpi.groupby('LOINC_CODE')['SUNQUEST_NAME'].apply(set).to_dict()

    rename_map = {}
    ambiguous = []

    for canon in canonical_names:
        loincs = name_to_loincs.get(canon)
        if not loincs:
            continue
        if len(loincs) > 1:
            ambiguous.append((canon, loincs))
            continue

        loinc = next(iter(loincs))
        for syn in loinc_to_names.get(loinc, set()):
            if syn == canon:
                continue
            syn_loincs = name_to_loincs.get(syn, set())
            if len(syn_loincs) > 1:
                ambiguous.append((syn, syn_loincs))
                continue
            rename_map[syn] = canon

    return rename_map, ambiguous


print("\nBuilding lab-name synonym map from 0data/dpi_lab_map.csv ...")
LAB_SYNONYM_MAP, AMBIGUOUS_NAMES = build_lab_synonym_map(
    '0data/dpi_lab_map.csv', BINS_THRESHOLDS.keys()
)
if LAB_SYNONYM_MAP:
    print(f"  Will normalize {len(LAB_SYNONYM_MAP)} synonym(s) into their canonical BINS_THRESHOLDS name:")
    for syn, canon in LAB_SYNONYM_MAP.items():
        print(f"    '{syn}' -> '{canon}'")
if AMBIGUOUS_NAMES:
    print(f"  WARNING: {len(AMBIGUOUS_NAMES)} name(s) have an AMBIGUOUS LOINC mapping "
          f"in dpi_lab_map.csv and were left untouched (review manually):")
    for name, loincs in AMBIGUOUS_NAMES:
        print(f"    '{name}' -> {sorted(loincs)}")


def sql_escape(s):
    return s.replace("'", "''")


# ---------------------------------------------------------------------------
# 1. Build the SQL CASE expression that normalizes synonym LAB_NAMEs into
#    their canonical BINS_THRESHOLDS key.
# ---------------------------------------------------------------------------
def build_rename_sql(rename_map, column="TEST_NAME"):
    if not rename_map:
        return column
    whens = " ".join(
        f"WHEN {column} = '{sql_escape(syn)}' THEN '{sql_escape(canon)}'"
        for syn, canon in rename_map.items()
    )
    return f"CASE {whens} ELSE {column} END"


# ---------------------------------------------------------------------------
# 2. Build the SQL CASE expression that does the binning (equivalent to
#    pd.cut with bins=[-inf, *thresholds, inf]), one nested CASE per test.
# ---------------------------------------------------------------------------
def build_binning_sql(bins_thresholds, name_column="TEST_NAME", value_column="TEST_VALUE"):
    outer_whens = []
    for test_name, thresholds in bins_thresholds.items():
        inner_whens = " ".join(
            f"WHEN {value_column} < {edge} THEN 'BIN_{i + 1}'"
            for i, edge in enumerate(thresholds)
        )
        inner_case = f"CASE {inner_whens} ELSE 'BIN_{len(thresholds) + 1}' END"
        outer_whens.append(f"WHEN {name_column} = '{sql_escape(test_name)}' THEN {inner_case}")
    return "CASE " + " ".join(outer_whens) + " ELSE NULL END"


RENAME_SQL = build_rename_sql(LAB_SYNONYM_MAP)
BINNING_SQL = build_binning_sql(BINS_THRESHOLDS)
# Only keep rows whose (post-rename) TEST_NAME is one we actually have
# thresholds for -- everything else is irrelevant to BEHRT and would just
# waste memory/disk if kept.
KNOWN_NAMES_SQL = ", ".join(f"'{sql_escape(n)}'" for n in BINS_THRESHOLDS.keys())

# ---------------------------------------------------------------------------
# 3. Run everything in DuckDB: read (streaming) -> clean -> rename synonyms
#    -> filter to known tests -> bin -> build the final CODE string.
#    Nothing is materialized in pandas until the very small final summary.
# ---------------------------------------------------------------------------
print("\nRunning DuckDB pipeline (this streams the CSVs, doesn't load them "
      "fully into Python memory)...")

con = duckdb.connect()
# DuckDB's own progress bar is what actually gives visibility into the heavy
# query below -- tqdm can't hook into a single atomic SQL COPY the way it
# could with the old row-by-row pandas loop, so we use DuckDB's native one
# instead (prints live progress to the terminal while the query runs).
con.execute("PRAGMA enable_progress_bar;")

query = f"""
    WITH Labs AS (
        SELECT
            PATIENT_ID,
            Shifted_date AS DATE,
            TRIM(LAB_NAME) AS TEST_NAME,
            TRY_CAST(RESULT AS DOUBLE) AS TEST_VALUE
        FROM read_csv_auto('0data/Deid_Lab_out.csv', ignore_errors=true)
    ),
    Vitals AS (
        SELECT
            PATIENT_ID,
            Shifted_date AS DATE,
            TRIM(REPLACE(VITAL_CODE, 'VITALS:', 'VITAL_')) || '_MEAN' AS TEST_NAME,
            TRY_CAST(MEASUREMENT AS DOUBLE) AS TEST_VALUE
        FROM read_csv_auto('0data/Deid_vital.csv', ignore_errors=true)
    ),
    Combined AS (
        SELECT * FROM Labs
        UNION ALL
        SELECT * FROM Vitals
    ),
    Cleaned AS (
        SELECT
            PATIENT_ID,
            DATE,
            {RENAME_SQL} AS TEST_NAME,
            TEST_VALUE
        FROM Combined
        WHERE TEST_VALUE IS NOT NULL
    ),
    Filtered AS (
        SELECT * FROM Cleaned
        WHERE TEST_NAME IN ({KNOWN_NAMES_SQL})
    ),
    Binned AS (
        SELECT
            PATIENT_ID,
            DATE,
            TEST_NAME,
            {BINNING_SQL} AS BINNED_VALUE
        FROM Filtered
    )
    SELECT
        PATIENT_ID,
        DATE,
        'LAB:' || TEST_NAME || '_' || BINNED_VALUE AS CODE
    FROM Binned
    WHERE BINNED_VALUE IS NOT NULL
"""

OUT_PATH = "0data/discretized_labs_vitals.csv"
con.execute(f"COPY ({query}) TO '{OUT_PATH}' (FORMAT CSV, HEADER TRUE);")

# Small summary query (safe -- COUNT/GROUP BY on the already-written result,
# doesn't require re-materializing the raw CSVs).
summary = con.execute(f"""
    SELECT
        regexp_extract(CODE, 'LAB:(.*)_BIN', 1) AS TEST_NAME,
        COUNT(*) AS n
    FROM read_csv_auto('{OUT_PATH}')
    GROUP BY 1
    ORDER BY n DESC
""").df()

print("\nRow counts per test after binning:")
print(summary.to_string(index=False))

print(f"\nSuccess! Labs and vitals discretized and saved as '{OUT_PATH}'")