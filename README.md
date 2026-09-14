# BEHRT: Early Prediction of Type 2 Diabetes from EHR Data

This repository contains a BERT style model (BEHRT) trained on longitudinal Electronic Health Record (EHR) data to predict the onset of Type 2 Diabetes before it is formally diagnosed. The project is part of a Master's thesis comparing this Transformer based approach against a traditional LSTM baseline.

## What the model does

Each patient's clinical history (diagnoses, labs, and vitals) is converted into a sequence of tokens, similar to how a sentence is a sequence of words. The model is pretrained with a masked language modeling objective on this vocabulary of clinical codes, then fine tuned to output the probability that a patient will develop Type 2 Diabetes, using only information available before their actual diagnosis date.

## Data and features

The dataset comes from a large, de identified EHR extract (roughly 289,000 patients). The model uses three feature types only:

* Diagnoses (ICD9/ICD10 codes)
* Labs and vitals (discretized into bins)
* Dynamic patient age at each event

Medications and demographic attributes (gender, race, ethnicity) are not used as model inputs. This was a deliberate scope decision, not an oversight, and is documented in the thesis as a possible direction for future work.

## Temporal censoring

A core requirement of this task is that the model must never see information from after a patient's diagnosis when predicting whether that patient will be diagnosed. All preprocessing scripts enforce this by keeping, for each diabetic patient, only events strictly before their earliest recorded diagnosis date, computed as the minimum across all diagnosis codes associated with that patient (not just the first row encountered, which would silently leak later events, see the comments in `01_build_universal_events.py` for the exact issue this avoids).

## Repository structure

```
1preprocessing/
    00_discretize_labs_vitals.py       Bins raw lab and vital measurements into tokens
    01_build_universal_events.py       Combines diagnoses and labs/vitals with temporal censoring
    02_format_for_finetuning.py        Builds per patient token sequences
    03_train_and_test_split.py         Stratified train/test split by patient
    04_generate_horizon_test_sets.py   Builds test sets censored 5, 3, 2, and 1 year before diagnosis
    generate_pretrain_data.py          Builds the self supervised pretraining corpus

2pretraining/
    MLM.ipynb                          Masked language model pretraining notebook

3finetuning/
    diabetes_fine_tuning_balanced.ipynb   Fine tuning notebook (10 iteration stability run)
    evaluate_and_shap.py               Evaluation, bootstrap confidence intervals, and SHAP interpretability
    evaluate_horizons_balanced.py      Reuses the 10 trained checkpoints on balanced horizon test sets
    evaluate_horizons_natural.py       Reuses the 10 trained checkpoints on natural prevalence horizon test sets

model/, dataLoader/, common/           Core BEHRT model and data loading code
```

All heavy data processing runs on DuckDB rather than pandas alone, since the raw event tables are large enough (over 80 million rows) that pandas and PyArrow repeatedly run out of memory on this hardware. Where a script mentions this, it is intentional, not a leftover experiment.

## Running the pipeline

1. Run the scripts inside `1preprocessing/` in numeric order (00 through 04), plus `generate_pretrain_data.py` for the pretraining corpus. Each script prints a short summary so you can sanity check patient and event counts as you go.
2. Run `2pretraining/MLM.ipynb` in Google Colab (GPU recommended) to pretrain the model on the masked language modeling task.
3. Run `3finetuning/diabetes_fine_tuning_balanced.ipynb` to fine tune the pretrained model. This performs 10 full training iterations, each with a fresh balanced sample and a fresh copy of the pretrained weights, and reports results as mean plus standard deviation, matching the methodology used for the LSTM baseline.
4. Use the scripts in `3finetuning/` to evaluate the saved checkpoints: overall performance, performance by prediction horizon, and SHAP based interpretability.

## Requirements

Python 3.10 or later, PyTorch, `pytorch_pretrained_bert`, DuckDB, pandas, scikit learn, and SHAP. A GPU is strongly recommended for both pretraining and fine tuning.

## Notes on methodology

The metrics reported for this model come in two flavors, and both matter:

* Balanced (1:1 undersampled) results, used for the head to head comparison against the LSTM baseline, since that is the only evaluation available on both sides.
* Natural prevalence results, evaluated on the real, imbalanced patient population, presented separately as the number that best reflects expected real world performance.

Model performance is also reported by prediction horizon (5, 3, 2, and 1 year before diagnosis), since predicting diabetes years in advance is a fundamentally harder and more clinically meaningful task than predicting it from data collected close to the diagnosis date.

## Known limitations and future work

The pretraining vocabulary is currently dominated by individual lab and vital readings, since every single measurement becomes its own token. A planned improvement is to aggregate labs and vitals to one value per patient per day before tokenizing, which should reduce sequence length for high utilization patients and give diagnosis codes more relative weight during pretraining. This has not yet been run and is left as a documented next step.
