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

# 1. Load your raw labs/vitals data 
# (Replace with your actual file path and make sure it contains PATIENT_ID, DATE, and the test columns)
df_labs = pd.read_csv('data/YOUR_LABS_AND_VITALS_DATA.csv') 

# 2. Convert from wide format to long format (melt)
df_long = df_labs.melt(id_vars=['PATIENT_ID', 'DATE'], var_name='TEST_NAME', value_name='TEST_VALUE')
df_long = df_long.dropna(subset=['TEST_VALUE']) # Remove null values

final_events = []

# 3. Apply the binning thresholds
for test_name, thresholds in BINS_THRESHOLDS.items():
    test_data = df_long[df_long['TEST_NAME'] == test_name].copy()
    if test_data.empty:
        continue
        
    # Add -infinity and +infinity to the thresholds to capture all possible values
    bins = [-np.inf] + thresholds + [np.inf]
    
    # Create the text labels (e.g., BIN_1, BIN_2, etc.)
    labels = [f"BIN_{i}" for i in range(1, len(bins))]
    
    # Categorize the numerical value into its corresponding bin
    test_data['BINNED_VALUE'] = pd.cut(test_data['TEST_VALUE'], bins=bins, labels=labels)
    
    # Construct the final discrete token string. Example: LAB:ALT(SGPT)_BIN_2
    test_data['CODE'] = 'LAB:' + test_data['TEST_NAME'] + '_' + test_data['BINNED_VALUE'].astype(str)
    
    final_events.append(test_data[['PATIENT_ID', 'DATE', 'CODE']])

# 4. Concatenate all processed events and save
df_all_labs = pd.concat(final_events)
df_all_labs.to_csv('data/discretized_labs_vitals.csv', index=False)
print("Success! Labs and vitals discretized and saved as 'data/discretized_labs_vitals.csv'")