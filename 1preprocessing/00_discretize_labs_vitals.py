import pandas as pd
import numpy as np

print("Starting Binning process for Labs and Vitals...")

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
    'VITAL_34506073_MEAN': [12, 21,31],
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
    'VITAL_14049161_MEAN': [40, 50, 100]
}

# 1. Load Labs: We use LAB_NAME as the test name to match the dictionary
print("Loading Labs...")
df_labs = pd.read_csv('0data/Deid_Lab_out.csv')
df_labs = df_labs[['PATIENT_ID', 'Shifted_date', 'LAB_NAME', 'RESULT']].rename(
    columns={'Shifted_date': 'DATE', 'LAB_NAME': 'TEST_NAME', 'RESULT': 'TEST_VALUE'}
)

# 2. Load Vitals: We use VITAL_CODE as the test name to match the dictionary
print("Loading Vitals...")
df_vitals = pd.read_csv('0data/Deid_vital.csv')
df_vitals = df_vitals[['PATIENT_ID', 'Shifted_date', 'VITAL_CODE', 'MEASUREMENT']].rename(
    columns={'Shifted_date': 'DATE', 'VITAL_CODE': 'TEST_NAME', 'MEASUREMENT': 'TEST_VALUE'}
)

# Format the Vitals string to match your dictionary exactly
df_vitals['TEST_NAME'] = df_vitals['TEST_NAME'].str.replace('VITALS:', 'VITAL_') + '_MEAN'

# 3. Combine both datasets into one master list
df_long = pd.concat([df_labs, df_vitals], ignore_index=True)

# Standardize text to avoid mismatch errors (remove trailing spaces)
df_long['TEST_NAME'] = df_long['TEST_NAME'].astype(str).str.strip()

# Clean up values (force to numbers, drop NaNs)
df_long['TEST_VALUE'] = pd.to_numeric(df_long['TEST_VALUE'], errors='coerce')
df_long = df_long.dropna(subset=['TEST_VALUE'])

final_events = []

# 4. Apply the binning thresholds
print("Applying discrete tokens...")
for test_name, thresholds in BINS_THRESHOLDS.items():
    test_data = df_long[df_long['TEST_NAME'] == test_name].copy()
    if test_data.empty:
        continue
        
    bins = [-np.inf] + thresholds + [np.inf]
    labels = [f"BIN_{i}" for i in range(1, len(bins))]
    
    test_data['BINNED_VALUE'] = pd.cut(test_data['TEST_VALUE'], bins=bins, labels=labels)
    
    # Construct the final discrete token string (e.g., "LAB:POTASSIUM_BIN_2")
    test_data['CODE'] = 'LAB:' + test_data['TEST_NAME'] + '_' + test_data['BINNED_VALUE'].astype(str)
    
    final_events.append(test_data[['PATIENT_ID', 'DATE', 'CODE']])

# 5. Save the final file for DuckDB to ingest
df_all_labs = pd.concat(final_events)
df_all_labs.to_csv('0data/discretized_labs_vitals.csv', index=False)
print("Success! Labs and vitals discretized and saved as '0data/discretized_labs_vitals.csv'")