import duckdb
con = duckdb.connect()

suspicious = con.execute("""
    SELECT PATIENT_ID, COUNT(*) AS n_suspicious_codes
    FROM read_parquet('0data/universal_events.parquet')
    WHERE CODE IN ('DX:ICD10CM:O24.113', 'DX:ICD10CM:Z79.4')
    GROUP BY PATIENT_ID
""").df()
print(f"Pacientes con estos códigos en su ventana pre-diagnóstico: {len(suspicious)}")

if len(suspicious) > 0:
    dx_dates = con.execute("SELECT * FROM read_csv_auto('0data/EARLIEST_dx_deid.csv')").df()
    example_id = suspicious['PATIENT_ID'].iloc[0]
    example_events = con.execute(f"""
        SELECT DATE, CODE FROM read_parquet('0data/universal_events.parquet')
        WHERE PATIENT_ID = '{example_id}' AND CODE IN ('DX:ICD10CM:O24.113', 'DX:ICD10CM:Z79.4')
        ORDER BY DATE
    """).df()
    print(f"\nPaciente ejemplo: {example_id}")
    print(f"Index_Date real: {dx_dates[dx_dates['PATIENT_ID']==example_id]}")
    print(example_events)