'''
import duckdb

con = duckdb.connect()
outliers = con.execute("""
    SELECT patid, len(code) AS seq_len
    FROM read_parquet('0data/behrt_finetuning_data.parquet')
    ORDER BY seq_len DESC
    LIMIT 20
""").df()
print(outliers)
'''

import duckdb

con = duckdb.connect()

PATID = "69361"  # ajusta el tipo (string/int) si tu PATIENT_ID no es string

# 1. Cuántos eventos brutos tiene este paciente, y en qué rango de fechas
summary = con.execute(f"""
    SELECT
        COUNT(*) AS total_events,
        COUNT(DISTINCT DATE) AS unique_dates,
        MIN(DATE) AS first_date,
        MAX(DATE) AS last_date,
        DATE_DIFF('year', MIN(DATE), MAX(DATE)) AS years_span
    FROM read_parquet('0data/universal_events.parquet')
    WHERE PATIENT_ID = '{PATID}'
""").df()
print("Resumen general:")
print(summary.to_string(index=False))

# 2. ¿Se concentran muchos eventos en el MISMO día? (señal de posible bug)
top_days = con.execute(f"""
    SELECT DATE, COUNT(*) AS n_events
    FROM read_parquet('0data/universal_events.parquet')
    WHERE PATIENT_ID = '{PATID}'
    GROUP BY DATE
    ORDER BY n_events DESC
    LIMIT 10
""").df()
print("\nDías con más eventos concentrados:")
print(top_days.to_string(index=False))

# 3. ¿Qué tipo de códigos predominan? (DX vs LAB/VITAL, y cuáles en concreto)
top_codes = con.execute(f"""
    SELECT CODE, COUNT(*) AS n
    FROM read_parquet('0data/universal_events.parquet')
    WHERE PATIENT_ID = '{PATID}'
    GROUP BY CODE
    ORDER BY n DESC
    LIMIT 15
""").df()
print("\nCódigos más frecuentes para este paciente:")
print(top_codes.to_string(index=False))