"""
04_generate_horizon_test_sets.py

Generates TEST SET versions censored at different prediction horizons
(5, 3, 2 and 1 year before the reference event), to evaluate how model
performance degrades the further out the prediction is made from diagnosis.

IMPORTANT:
- The TRAIN SET is not touched here. It keeps using the full pre-diagnosis
  history, as produced by 03_train_and_test_split.py.
- The test-set patient population is ALWAYS the same across all 4 horizons
  (comes from 0data/diabetes_test.parquet). Only how much of each patient's
  history the model gets to see changes.

Censoring rule per horizon:
- Cases (diabetic):     cutoff = Index_Date (EARLIEST_DX) - N years
- Controls (non-diab.): cutoff = last visit date in their history (already
                         censored at the universal_events.parquet level) - N years
  -> This is "Option A" (simpler). Documented as a methodological
     limitation: unlike case-control matching by index date (Option C),
     controls with shorter/discontinuous histories may lose more patients
     when the cutoff is applied, which can bias the surviving control
     sample towards patients with longer follow-up. Mention this explicitly
     in the thesis limitations section.
"""

import pandas as pd
import numpy as np
import duckdb
import os

HORIZONS_YEARS = [5, 3, 2, 1]

BEHRT_ROOT = "/content/drive/MyDrive/Colab Notebooks/BEHRT"
DATA_DIR = os.path.join(BEHRT_ROOT, "0data")
OUT_DIR = DATA_DIR


def read_parquet_safe(path):
    """Plain pd.read_parquet() can throw
    'ArrowNotImplementedError: Nested data conversions not implemented for
    chunked array outputs' on parquet files with large list-type columns
    (our 'code'/'age' sequences -- some patients have 60k+ tokens even after
    the leakage fix). DuckDB reads and converts these to pandas without
    hitting that limitation."""
    con = duckdb.connect()
    return con.execute(f"SELECT * FROM read_parquet('{path}')").df()


print("=" * 60)
print(" GENERATING HORIZON-CENSORED TEST SETS (5y / 3y / 2y / 1y) ")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Fixed test-set population (same patients across all horizons)
# ---------------------------------------------------------------------------
print("\n[1/5] Loading fixed test-set patient IDs and labels...")
df_test_base = read_parquet_safe(os.path.join(DATA_DIR, 'diabetes_test.parquet'))
test_patids = set(df_test_base['patid'].unique())
labels = df_test_base.set_index('patid')['label'].to_dict()
print(f"      Test set has {len(test_patids)} patients "
      f"({sum(v == 1 for v in labels.values())} cases / "
      f"{sum(v == 0 for v in labels.values())} controls).")

# ---------------------------------------------------------------------------
# 2. Already-censored universal events (DX + LABS + VITALS, censored at
#    Index_Date for cases, per 01_build_universal_events.py), restricted to
#    test patients only.
# ---------------------------------------------------------------------------
print("\n[2/5] Loading censored universal events for test patients...")
events_df = pd.read_parquet(os.path.join(DATA_DIR, 'universal_events.parquet'))
events_df = events_df[events_df['PATIENT_ID'].isin(test_patids)].copy()
events_df['DATE'] = pd.to_datetime(events_df['DATE'])
events_df['CODE'] = events_df['CODE'].astype(str)
print(f"      {events_df['PATIENT_ID'].nunique()} test patients found with events.")

# ---------------------------------------------------------------------------
# 3. Reference dates: Index_Date for cases, last visit date for controls
# ---------------------------------------------------------------------------
print("\n[3/5] Computing per-patient reference dates...")
dx_dates = pd.read_csv(os.path.join(DATA_DIR, 'EARLIEST_DX_deid.csv'))
dx_dates['EARLIEST_DX'] = pd.to_datetime(dx_dates['EARLIEST_DX'])
# CRITICAL FIX: EARLIEST_dx_deid.csv has one row per (PATIENT_ID, DX) pair,
# not one row per patient -- each distinct diagnosis code has its own
# "first occurrence" date. Without MIN()+GROUP BY, .set_index().to_dict()
# would just keep whichever row happens to be last for a given patient (not
# necessarily their true earliest diagnosis) -- must aggregate with min()
# per patient first, same fix applied in 01_build_universal_events.py and
# generate_pretrain_data.py.
dx_dates = dx_dates.groupby('PATIENT_ID')['EARLIEST_DX'].min().reset_index()
index_date_map = dx_dates.set_index('PATIENT_ID')['EARLIEST_DX'].to_dict()

last_visit_map = events_df.groupby('PATIENT_ID')['DATE'].max().to_dict()


def reference_date(pid):
    if pid in index_date_map:
        return index_date_map[pid]           # cases: real diagnosis date
    return last_visit_map.get(pid, pd.NaT)    # controls: their own last visit


ref_date_map = {pid: reference_date(pid) for pid in test_patids}
n_missing_ref = sum(pd.isna(d) for d in ref_date_map.values())
if n_missing_ref:
    print(f"      WARNING: {n_missing_ref} test patients have no events at all "
          f"and will be dropped from every horizon.")

# ---------------------------------------------------------------------------
# 4. Stable per-patient age (YOB) using the FULL pre-diagnosis censored
#    history, so ages stay consistent with the original (non-horizon) sets.
# ---------------------------------------------------------------------------
print("\n[4/5] Computing stable year-of-birth per patient (for AGE tokens)...")
demo_df = pd.read_csv(os.path.join(DATA_DIR, 'deid_DEM.csv'))[['PATIENT_ID', 'AGE_AT_END']]
last_visit_df = (events_df.groupby('PATIENT_ID')['DATE'].max()
                  .reset_index().rename(columns={'DATE': 'LAST_VISIT_DATE'}))
patient_info = last_visit_df.merge(demo_df, on='PATIENT_ID', how='inner')
patient_info['YOB'] = patient_info['LAST_VISIT_DATE'].dt.year - patient_info['AGE_AT_END']
yob_map = patient_info.set_index('PATIENT_ID')['YOB'].to_dict()

events_df['YOB'] = events_df['PATIENT_ID'].map(yob_map)
events_df = events_df.dropna(subset=['YOB'])
events_df['AGE'] = (events_df['DATE'].dt.year - events_df['YOB']).clip(lower=0).astype(int).astype(str)
events_df = events_df.sort_values(['PATIENT_ID', 'DATE'])


def build_patient_sequence(group):
    """Flatten a patient's (already horizon-truncated) events into a single
    code/age sequence, inserting 'SEP' at the end of each visit (same day)."""
    codes, ages = [], []
    for _, visit in group.groupby('DATE'):
        codes.extend(visit['CODE'].tolist() + ['SEP'])
        visit_ages = visit['AGE'].tolist()
        visit_ages.append(visit_ages[-1])
        ages.extend(visit_ages)
    return pd.Series({'code': codes, 'age': ages})


# ---------------------------------------------------------------------------
# 5. Build one censored test set per horizon
# ---------------------------------------------------------------------------
print("\n[5/5] Building horizon-specific test sets...")
for years in HORIZONS_YEARS:
    print(f"\n--- {years}-year horizon ---")
    cutoff_map = {pid: (ref - pd.DateOffset(years=years))
                  for pid, ref in ref_date_map.items() if pd.notna(ref)}

    ev = events_df.copy()
    ev['CUTOFF'] = ev['PATIENT_ID'].map(cutoff_map)
    ev = ev.dropna(subset=['CUTOFF'])
    ev = ev[ev['DATE'] < ev['CUTOFF']]

    if ev.empty:
        print(f"  No events survive the {years}-year cutoff. Skipping.")
        continue

    seqs = ev.groupby('PATIENT_ID').apply(build_patient_sequence).reset_index()
    seqs = seqs.rename(columns={'PATIENT_ID': 'patid'})

    # Same minimum-context filter used everywhere else in the pipeline
    seqs = seqs[seqs['code'].apply(len) >= 2]

    seqs['label'] = seqs['patid'].map(labels)
    seqs = seqs.dropna(subset=['label'])
    seqs['label'] = seqs['label'].astype(int)

    out_path = os.path.join(OUT_DIR, f'diabetes_test_{years}y.parquet')
    seqs.to_parquet(out_path, index=False)

    n_total = len(seqs)
    n_pos = int(seqs['label'].sum())
    n_dropped = len(test_patids) - n_total
    print(f"  Kept {n_total}/{len(test_patids)} patients "
          f"({n_pos} cases / {n_total - n_pos} controls). "
          f"Dropped {n_dropped} (insufficient history at this horizon).")
    print(f"  Saved -> {out_path}")

print("\n" + "=" * 60)
print(" DONE. Horizon test sets ready for evaluation / SHAP.")
print(" Remember: sample size will shrink as the horizon grows (5y < 1y),")
print(" this is expected -- fewer patients have >=2 years of pre-diagnosis")
print(" history 5 years out than 1 year out.")
print("=" * 60)