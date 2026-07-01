import os
import glob
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

print("Loading CENSORSED universal events and demographics for Fine-Tuning...")
events_df = pd.read_parquet('data/universal_events.parquet')
demo_df = pd.read_csv('data/deid_DEM.csv')

# Ensure correct data types
events_df['DATE'] = pd.to_datetime(events_df['DATE'])
events_df['CODE'] = events_df['CODE'].astype(str)

print("Reverse engineering to calculate dynamic age for each visit...")

# 1. Find the LAST visit date for each patient
last_visits = events_df.groupby('PATIENT_ID')['DATE'].max().reset_index()
last_visits = last_visits.rename(columns={'DATE': 'LAST_VISIT_DATE'})

# 2. Merge this last date with their AGE_AT_END
demo_subset = demo_df[['PATIENT_ID', 'AGE_AT_END']]
patient_info = last_visits.merge(demo_subset, on='PATIENT_ID', how='inner')

# 3. CALCULATE YEAR OF BIRTH (YOB)
patient_info['YOB'] = patient_info['LAST_VISIT_DATE'].dt.year - patient_info['AGE_AT_END']

# 4. Now merge this YOB back to ALL events for the patient
df = events_df.merge(patient_info[['PATIENT_ID', 'YOB']], on='PATIENT_ID', how='inner')

# 5. Calculate the exact DYNAMIC age for each specific event
df['AGE'] = df['DATE'].dt.year - df['YOB']

# Ensure there are no negative ages due to potential data errors
df['AGE'] = df['AGE'].apply(lambda x: max(0, x))

# BEHRT requires age to be a string format for its vocabulary
df['AGE'] = df['AGE'].astype(str) 

print("Sorting chronologically...")
df = df.sort_values(by=['PATIENT_ID', 'DATE'])

def process_visit(group):
    visit_codes = group['CODE'].tolist() + ['SEP']
    visit_ages = group['AGE'].tolist()
    if len(visit_ages) > 0:
        visit_ages.append(visit_ages[-1]) 
    return pd.Series({'code': visit_codes, 'age': visit_ages})

print("\n--- STARTING BATCH PROCESSING (CHECKPOINTS) ---")

# 2. CAMBIO: Usamos carpetas temporales nuevas para no mezclar datos
os.makedirs('data/temp_ft_visits_chunks', exist_ok=True)

unique_patients = df['PATIENT_ID'].unique()
chunk_size = 5000 
total_chunks = (len(unique_patients) // chunk_size) + 1

for i in tqdm(range(total_chunks), desc="Grouping Visits"):
    chunk_file = f'data/temp_ft_visits_chunks/visit_chunk_{i}.parquet'
    
    if os.path.exists(chunk_file):
        continue
        
    batch_ids = unique_patients[i*chunk_size : (i+1)*chunk_size]
    batch_df = df[df['PATIENT_ID'].isin(batch_ids)]
    
    visits_batch = batch_df.groupby(['PATIENT_ID', 'DATE']).apply(process_visit).reset_index()
    visits_batch.to_parquet(chunk_file)

print("\nVisit grouping completed! Merging all batches...")

all_visit_files = glob.glob('data/temp_ft_visits_chunks/visit_chunk_*.parquet')
visits_df = pd.concat([pd.read_parquet(f) for f in all_visit_files])

print("\n--- FLATTENING SEQUENCES (CHECKPOINTS) ---")
os.makedirs('data/temp_ft_patient_chunks', exist_ok=True)

def process_patient(group):
    patient_codes = [code for visit in group['code'] for code in visit]
    patient_ages = [age for visit in group['age'] for age in visit]
    return pd.Series({'code': patient_codes, 'age': patient_ages})

for i in tqdm(range(total_chunks), desc="Flattening Patients"):
    chunk_file = f'data/temp_ft_patient_chunks/patient_chunk_{i}.parquet'
    
    if os.path.exists(chunk_file):
        continue
        
    batch_ids = unique_patients[i*chunk_size : (i+1)*chunk_size]
    batch_df = visits_df[visits_df['PATIENT_ID'].isin(batch_ids)]
    
    patient_batch = batch_df.groupby('PATIENT_ID').apply(process_patient).reset_index()
    patient_batch.to_parquet(chunk_file)

print("\nFlattening completed! Generating final file for Fine-Tuning...")

all_patient_files = glob.glob('data/temp_ft_patient_chunks/patient_chunk_*.parquet')
final_behrt_df = pd.concat([pd.read_parquet(f) for f in all_patient_files])

final_behrt_df = final_behrt_df.rename(columns={'PATIENT_ID': 'patid'})

# 4. CAMBIO MÁS IMPORTANTE: Guardamos con un nombre distinto
final_behrt_df.to_parquet('data/behrt_finetuning_data.parquet')

print("Absolute success! Your sequence data is ready for the Train/Test split.")