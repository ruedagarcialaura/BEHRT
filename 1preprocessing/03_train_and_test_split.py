import pandas as pd
import duckdb
from sklearn.model_selection import train_test_split

print("1. Loading censored sequences for Fine-Tuning...")
df_seq = pd.read_parquet('data/behrt_finetuning_data.parquet')

print("2. Extracting AIM_GROUP label from the database...")
# Use DuckDB to quickly extract the ID -> AIM_GROUP mapping without loading the giant CSV into memory
query = """
    SELECT DISTINCT PATIENT_ID, AIM_GROUP 
    FROM read_csv_auto('data/deid_visit_dx.csv')
"""
df_labels = duckdb.query(query).df()

print("3. Merging data and mapping labels...")
# Merge sequences with their final label
df_final = pd.merge(df_seq, df_labels, left_on='patid', right_on='PATIENT_ID', how='inner')

# Convert text to binary numbers for the neural network
df_final['label'] = df_final['AIM_GROUP'].map({'1_NoD': 0, '2_Type2': 1})
df_final = df_final.dropna(subset=['label'])
df_final['label'] = df_final['label'].astype(int)

# IMPORTANT: Due to censoring, some diabetic patients might end up with 0 previous visits.
# We filter to keep only patients with at least 2 codes in their history
# so the model actually has some context to read.
df_final = df_final[df_final['code'].apply(len) >= 2]

print("4. Splitting into Train (80%) and Test (20%)...")
# Stratified split to maintain the same proportion of diabetics in both sets
df_train, df_test = train_test_split(
    df_final, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_final['label']
)

# Save the final files for Colab
df_train.to_parquet('data/diabetes_train.parquet', index=False)
df_test.to_parquet('data/diabetes_test.parquet', index=False)

print("--------------------------------------------------")
print("DATA PREPARATION COMPLETED!")
print(f"Total patients for Training (Train): {len(df_train)}")
print(f"Total patients for Testing (Test): {len(df_test)}")
print("\nLabel distribution in Train:")
print(df_train['label'].value_counts())
print("--------------------------------------------------")
print("Next stop: Google Colab for Fine-Tuning!")