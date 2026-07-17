import duckdb
import pandas as pd
import os
import glob
from tqdm import tqdm

print("==================================================")
print(" PHASE 1: EXTRACTING FULL UNCENSORED HISTORY ")
print("==================================================")

# 1. Connect to DuckDB
con = duckdb.connect()

# Combine Diagnoses and Labs/Vitals (NO MEDICATIONS). 
# Notice there is NO 'WHERE' clause here to censor dates. We want the full history!
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
    )
    SELECT * FROM Diagnosis
    UNION ALL
    SELECT * FROM LabsVitals
"""

print("Executing DuckDB query to combine all events...")
con.execute(f"COPY ({query}) TO '1preprocessing/preprocessed_data/temp_uncensored_events.parquet' (FORMAT PARQUET);")


print("\n==================================================")
print(" PHASE 2: BUILDING LONGITUDINAL SEQUENCES ")
print("==================================================")

events_df = pd.read_parquet('1preprocessing/preprocessed_data/temp_uncensored_events.parquet')
demo_df = pd.read_csv('0data/deid_DEM.csv')

events_df['DATE'] = pd.to_datetime(events_df['DATE'])
events_df['CODE'] = events_df['CODE'].astype(str)

print("Calculating dynamic age for each visit...")
# Find the LAST visit date for each patient to calculate their birth year
last_visits = events_df.groupby('PATIENT_ID')['DATE'].max().reset_index()
last_visits = last_visits.rename(columns={'DATE': 'LAST_VISIT_DATE'})

demo_subset = demo_df[['PATIENT_ID', 'AGE_AT_END']]
patient_info = last_visits.merge(demo_subset, on='PATIENT_ID', how='inner')

# Calculate Year of Birth (YOB)
patient_info['YOB'] = patient_info['LAST_VISIT_DATE'].dt.year - patient_info['AGE_AT_END']

# Merge YOB back to all events and calculate specific age per event
df = events_df.merge(patient_info[['PATIENT_ID', 'YOB']], on='PATIENT_ID', how='inner')
df['AGE'] = df['DATE'].dt.year - df['YOB']
df['AGE'] = df['AGE'].apply(lambda x: max(0, x)).astype(str) # No negative ages, string format for BEHRT

print("Sorting chronologically...")
df = df.sort_values(by=['PATIENT_ID', 'DATE'])

def process_visit(group):
    visit_codes = group['CODE'].tolist() + ['SEP']
    visit_ages = group['AGE'].tolist()
    if len(visit_ages) > 0:
        visit_ages.append(visit_ages[-1]) 
    return pd.Series({'code': visit_codes, 'age': visit_ages})

print("Batch processing visits (this might take a minute)...")
os.makedirs('1preprocessing/preprocessed_data/temp_pretrain_visits', exist_ok=True)
unique_patients = df['PATIENT_ID'].unique()
chunk_size = 5000 
total_chunks = (len(unique_patients) // chunk_size) + 1

for i in tqdm(range(total_chunks), desc="Grouping Visits"):
    chunk_file = f'1preprocessing/preprocessed_data/temp_pretrain_visits/visit_chunk_{i}.parquet'
    if not os.path.exists(chunk_file):
        batch_ids = unique_patients[i*chunk_size : (i+1)*chunk_size]
        batch_df = df[df['PATIENT_ID'].isin(batch_ids)]
        visits_batch = batch_df.groupby(['PATIENT_ID', 'DATE']).apply(process_visit).reset_index()
        visits_batch.to_parquet(chunk_file)

all_visit_files = glob.glob('1preprocessing/preprocessed_data/temp_pretrain_visits/visit_chunk_*.parquet')
visits_df = pd.concat([pd.read_parquet(f) for f in all_visit_files])

print("Flattening sequences for each patient...")
os.makedirs('1preprocessing/preprocessed_data/temp_pretrain_patients', exist_ok=True)

def process_patient(group):
    patient_codes = [code for visit in group['code'] for code in visit]
    patient_ages = [age for visit in group['age'] for age in visit]
    return pd.Series({'code': patient_codes, 'age': patient_ages})

for i in tqdm(range(total_chunks), desc="Flattening Patients"):
    chunk_file = f'1preprocessing/preprocessed_data/temp_pretrain_patients/patient_chunk_{i}.parquet'
    if not os.path.exists(chunk_file):
        batch_ids = unique_patients[i*chunk_size : (i+1)*chunk_size]
        batch_df = visits_df[visits_df['PATIENT_ID'].isin(batch_ids)]
        patient_batch = batch_df.groupby('PATIENT_ID').apply(process_patient).reset_index()
        patient_batch.to_parquet(chunk_file)

all_patient_files = glob.glob('1preprocessing/preprocessed_data/temp_pretrain_patients/patient_chunk_*.parquet')
final_behrt_df = pd.concat([pd.read_parquet(f) for f in all_patient_files])

final_behrt_df = final_behrt_df.rename(columns={'PATIENT_ID': 'patid'})

# Filter out patients with less than 2 codes (the model needs context to learn)
final_behrt_df = final_behrt_df[final_behrt_df['code'].apply(len) >= 2]

final_behrt_df.to_parquet('1preprocessing/preprocessed_data/behrt_pretrain_data.parquet', index=False)

print("\n==================================================")
print(" SUCCESS! PRE-TRAINING DATA READY ")
print(" Saved as: '1preprocessing/preprocessed_data/behrt_pretrain_data.parquet' ")
print("==================================================")