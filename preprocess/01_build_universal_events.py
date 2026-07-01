import duckdb

# 1. Connect to DuckDB
con = duckdb.connect()

print("Fusing Diagnoses and Medications and applying Temporal Censoring (Data Leakage Prevention)...")

# 2. Advanced SQL query with data leakage prevention
query = """
    -- Step A: Obtain Diagnoses
    WITH Diagnosis AS (
        SELECT 
            PATIENT_ID, 
            CAST(Shifted_date AS DATE) AS DATE, 
            'DX:' || CAST(DX AS VARCHAR) AS CODE 
        FROM read_csv_auto('data/deid_visit_dx.csv')
    ),
    
    -- Step B: Obtain Medications
    Medications AS (
        SELECT 
            PATIENT_ID, 
            CAST(Shifted_date AS DATE) AS DATE, 
            'RX:' || CAST(RX_CODE AS VARCHAR) AS CODE 
        FROM read_csv_auto('data/deid_rx_order.csv')
    ),
    
    -- Step C: Combine all events (The original Universal Events)
    Universal_Events AS (
        SELECT * FROM Diagnosis
        UNION ALL
        SELECT * FROM Medications
    ),
    
    -- Step D: Load the exact diabetes diagnosis dates
    Diabetes_dates AS (
        SELECT 
            PATIENT_ID, 
            CAST(EARLIEST_DX AS DATE) AS Index_Date
        FROM read_csv_auto('data/EARLIEST_dx_deid.csv')
    )
    
    -- Step E: Final Filtering (Temporal Censoring)
    SELECT 
        u.PATIENT_ID, 
        u.DATE, 
        u.CODE
    FROM Universal_Events u
    LEFT JOIN Diabetes_dates f ON u.PATIENT_ID = f.PATIENT_ID
    -- GOLDEN RULE:
    -- If the patient is not in the diabetes table (f.Index_Date IS NULL), include all events.
    -- If the patient is diabetic, only include events BEFORE their diagnosis (<)
    WHERE f.Index_Date IS NULL OR u.DATE < f.Index_Date
"""

# 3. Run and save
con.execute(f"COPY ({query}) TO 'data/universal_events.parquet' (FORMAT PARQUET);")

print("Success! Cleaned and censored events saved to 'data/universal_events.parquet'.")