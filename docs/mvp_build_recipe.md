# Exact MVP build recipe

This is the canonical step-by-step build plan for the PAMAP2 wearable telemetry MVP. It should stay available to Copilot and to future work sessions as the main implementation recipe.

This recipe is intentionally practical. The point is to finish a strong project quickly without unnecessary scope creep.

---

# 1. Final MVP definition

## 1.1. Project goal

Build a wearable telemetry modeling pipeline from PAMAP2 that does three things from one shared dataset pipeline:

1. forecast a future numeric signal,
2. classify an activity state,
3. estimate uncertainty.

## 1.2. Chosen MVP tasks

### 1.2.1. Regression task
Predict heart rate 30 seconds ahead from recent wearable sensor history.

### 1.2.2. Classification task
Predict the current activity label from a recent sensor window.

### 1.2.3. Uncertainty task
Produce prediction intervals for the heart-rate forecast using a lightweight conformal method.

## 1.3. Why this is the right MVP

This gives one coherent story instead of several disconnected mini-projects:

- telemetry ingestion and cleaning
- time-series feature engineering
- forecasting
- state classification
- uncertainty estimation
- proper evaluation

## 1.4. What not to build in v1

Do not add these in the MVP:

- API
- Streamlit
- cloud
- deep learning
- Dask unless the notebook is actually slow
- fancy experiment tracking
- dozens of models
- exhaustive hyperparameter tuning

The MVP wins by being clean, correct, and complete.

---

# 2. Final deliverables by the end of the MVP

## 2.1. One project folder

With:
- raw data
- processed tables
- notebooks
- saved figures
- saved metrics tables

## 2.2. One strong modeling table

A clean per-time-step feature table with:
- subject ID
- timestamp
- activity label
- heart rate
- sensor summary features
- lagged and rolling features
- future heart-rate target

## 2.3. One finished notebook story

It should clearly show:
- raw data audit
- preprocessing decisions
- feature engineering
- split strategy
- regression results
- classification results
- uncertainty interval results
- final conclusions

## 2.4. One short written project summary

A short README or markdown section answering:
- what problem was solved
- how the data flows
- what models were tried
- how they were evaluated
- what worked
- what would be built next

---

# 3. Final project structure

## 3.1. Folder layout

pamap2_telemetry_mvp/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_ingest_and_audit.ipynb
│   ├── 02_feature_pipeline.ipynb
│   └── 03_modeling_and_uncertainty.ipynb
│
├── artifacts/
│   ├── figures/
│   ├── metrics/
│   └── models/
│
├── README.md
└── requirements.txt

## 3.2. Why this structure

It is simple, readable, and interview-safe. It shows organization without overengineering.

---

# 4. Exact MVP modeling design

## 4.1. Core prediction tasks

### 4.1.1. Regression task
Target:
- heart rate at `t + 30 seconds`

### 4.1.2. Classification task
Target:
- current activity state at time `t`

### 4.1.3. Uncertainty task
Build prediction intervals for the heart-rate forecast using split conformal prediction on top of the final regression model.

---

## 4.2. Modeling granularity

### 4.2.1. Raw sensor data is too fine for the MVP
Do not model directly from full raw high-frequency signals end to end.

### 4.2.2. Build a 1-second telemetry table
Resample or aggregate the raw signals into 1-second rows per subject.

Each row should represent what the wearable system knows at second `t`.

---

## 4.3. Feature design

## 4.3.1. Use a compact set of sensor summaries
Do not explode the feature space too early.

Start with these groups:
- heart rate
- accelerometer magnitude by body location
- gyroscope magnitude by body location
- optional temperature by body location

## 4.3.2. For each body location
For hand, chest, and ankle:
- compute accelerometer magnitude
- compute gyroscope magnitude

Formula:
- magnitude = sqrt(x^2 + y^2 + z^2)

This gives interpretable telemetry features without drowning in columns.

---

## 4.4. Rolling-window features

## 4.4.1. Build features from the recent past only
For each 1-second row, create features using the previous 10 seconds.

## 4.4.2. For each numeric signal, create:
- current value
- lag 1 second
- lag 5 seconds
- rolling mean over last 5 seconds
- rolling std over last 5 seconds
- rolling mean over last 10 seconds
- rolling std over last 10 seconds
- short-term change: current minus 5-second rolling mean

## 4.4.3. For the activity label
Use the current activity label as the classification target. Do not use the future label anywhere in feature creation.

---

## 4.5. Activity scope

## 4.5.1. Do not force all activities into v1
Use only the core activity classes with enough data.

Good v1 rule:
- inspect class counts first
- keep the major activities
- drop rare or noisy labels if needed

## 4.5.2. Likely good v1 activity set
A likely clean set is:
- lying
- sitting
- standing
- walking
- running
- cycling

This is only a default guess. Final choice should be based on the audit.

---

# 5. Exact evaluation design

## 5.1. Split strategy

### 5.1.1. Do not randomly split rows
Random row-level splitting creates leakage because neighboring time points are too similar.

### 5.1.2. Split by subject first
Use a subject-based split.

Example:
- train subjects: 1 to 6
- validation subject: 7
- test subjects: 8 to 9

The exact IDs can change. The rule is:
- new people in the test set

---

## 5.2. Regression metrics
Use:
- MAE
- RMSE
- optional R^2

For intervals, use:
- empirical coverage
- average interval width

## 5.3. Classification metrics
Use:
- accuracy
- macro F1
- confusion matrix

Macro F1 matters because classes may be imbalanced.

---

# 6. Exact build recipe

# Phase 0 — Setup and contract

## 0.1. Create the project folder
Create the folder structure shown earlier.

## 0.2. Create the environment
Install only what you need:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pyarrow
- jupyter

Optional later:
- Dask

## 0.3. Create the notebooks
Create:
- `01_ingest_and_audit.ipynb`
- `02_feature_pipeline.ipynb`
- `03_modeling_and_uncertainty.ipynb`

## 0.4. Write the project contract
At the top of notebook 1, write a markdown block with:
- problem statement
- regression target
- classification target
- uncertainty method
- split rule
- what is out of scope

### Definition of done for Phase 0
You have:
- folders
- environment
- empty notebooks
- one-paragraph project contract

---

# Phase 1 — Ingest and audit PAMAP2

## 1.1. Download the dataset
Put the raw files in `data/raw/`.

## 1.2. Read the dataset documentation carefully
Before doing anything else:
- identify the timestamp column
- identify subject files
- identify activity label column
- identify heart-rate column
- identify body-location sensor columns

## 1.3. Build a small data dictionary
In notebook 1, make a table with:
- raw column name
- meaning
- keep or drop
- reason

## 1.4. Load one subject first
Do not load everything immediately.
Load one subject and verify:
- row count
- column count
- missing values
- activity label values
- heart-rate missingness
- timestamp behavior

## 1.5. Then load all subjects
Concatenate them into one dataframe with:
- `subject_id`
- `timestamp`
- raw sensor columns
- `activity_id`
- `heart_rate`

## 1.6. Run the audit
Produce the following outputs:
- rows per subject
- missingness by column
- activity counts overall
- activity counts by subject
- summary stats for heart rate
- a quick time plot of heart rate for one subject
- a quick time plot of one accelerometer magnitude for one subject

## 1.7. Make early cleaning decisions
Document:
- which activities to keep
- how invalid or unlabeled rows are treated
- how missing heart rate is treated

### Definition of done for Phase 1
You have:
- one combined raw dataframe
- a clear activity set for v1
- a clear keep or drop column list
- audit plots and summary tables

---

# Phase 2 — Build the 1-second telemetry table

## 2.1. Compute sensor magnitudes
For each body location:
- accelerometer magnitude
- gyroscope magnitude

## 2.2. Keep the compact telemetry feature set
The working columns should be something like:
- subject_id
- timestamp
- activity_id
- heart_rate
- hand_acc_mag
- chest_acc_mag
- ankle_acc_mag
- hand_gyro_mag
- chest_gyro_mag
- ankle_gyro_mag
- optional temperatures

## 2.3. Convert timestamps to a consistent second-level grid
Group by subject and resample to 1-second intervals.

## 2.4. Aggregate within each second
For each numeric signal:
- use the mean within the second

For activity:
- use the modal or majority activity label within the second

## 2.5. Handle missing heart rate after resampling
Use a simple, documented choice:
- forward fill within subject
- optionally backfill only short gaps
- no complex imputation in v1

## 2.6. Save the per-second telemetry table
Write it to:
- `data/interim/pamap2_per_second.parquet`

### Definition of done for Phase 2
You have one clean per-second table with one row per subject-second.

---

# Phase 3 — Build lagged and rolling features

## 3.1. Sort properly
Sort by:
- subject_id
- timestamp

Always do this before rolling or lagging.

## 3.2. Generate past-only features
Within each subject, for each numeric telemetry signal, create:
- lag_1
- lag_5
- rollmean_5
- rollstd_5
- rollmean_10
- rollstd_10
- delta_from_rollmean_5

## 3.3. Define the regression target
Create:
- `hr_target_30s = heart_rate shifted by -30 seconds within subject`

If that target is too sparse or noisy, fallback to:
- mean heart rate over the next 30 seconds

Try the simple shift first.

## 3.4. Define the classification target
Use:
- `activity_target = current activity_id`

## 3.5. Drop rows made invalid by windowing
After creating lagged and future targets:
- drop rows with insufficient past window
- drop rows without future target
- drop rows with missing essential features

## 3.6. Save the final modeling table
Write to:
- `data/processed/pamap2_model_table.parquet`

### Definition of done for Phase 3
You have one final supervised-learning table.

---

# Phase 4 — Exploratory analysis on the modeling table

## 4.1. Regression EDA
Plot:
- heart rate distribution
- heart rate by activity
- example heart-rate time series for one subject
- correlation heatmap for top telemetry features versus heart-rate target

## 4.2. Classification EDA
Plot:
- class balance
- example sensor traces by activity
- activity-specific distributions of motion features

## 4.3. Sanity-check target difficulty
Before modeling, verify:
- heart rate 30 seconds ahead is still related to recent heart rate and motion
- activities are separable enough to be worth classifying

### Definition of done for Phase 4
There is evidence the chosen targets are learnable for the chosen scope.

---

# Phase 5 — Build the train, validation, and test split

## 5.1. Split by subject
Make three separate tables:
- train
- validation
- test

## 5.2. Freeze the split
Do not keep changing it unless something is clearly broken.

## 5.3. Separate features and targets
Create:
- `X_train_reg`, `y_train_reg`
- `X_val_reg`, `y_val_reg`
- `X_test_reg`, `y_test_reg`

And similarly for classification.

### Definition of done for Phase 5
All downstream modeling uses the same split.

---

# Phase 6 — Baseline models first

## 6.1. Regression baselines
Build:
- naive baseline: predict current heart rate
- rolling baseline: predict 10-second rolling mean heart rate
- linear regression baseline

## 6.2. Classification baselines
Build:
- majority-class baseline
- logistic regression baseline
- optional shallow tree baseline

## 6.3. Record results in one metrics table
Make one clean dataframe with:
- model name
- task
- split
- metrics

### Definition of done for Phase 6
You have honest baselines to beat.

---

# Phase 7 — Stronger models

## 7.1. Regression
Use one stronger sklearn model:
- `HistGradientBoostingRegressor`
or
- `RandomForestRegressor`

Default choice:
- `HistGradientBoostingRegressor`

## 7.2. Classification
Use one stronger sklearn model:
- `RandomForestClassifier`
or
- `HistGradientBoostingClassifier`

Default choice:
- `RandomForestClassifier`

## 7.3. Do minimal tuning only
Tune only a few parameters:
- max depth
- number of estimators or learning rate
- min samples leaf

Keep this lean.

## 7.4. Compare against baselines
If the stronger model does not beat the baseline meaningfully, do not hide that. Explain it.

### Definition of done for Phase 7
You have one best regression model and one best classification model.

---

# Phase 8 — Add uncertainty to the regression model

## 8.1. Use split conformal prediction
Do not overcomplicate this.

Procedure:
1. fit the final regression model on train
2. predict on validation
3. compute absolute residuals on validation
4. choose the residual quantile for the desired coverage
5. apply that margin to test predictions

## 8.2. Produce intervals
For each test prediction:
- lower bound
- point prediction
- upper bound

## 8.3. Evaluate interval quality
Report:
- empirical coverage
- average interval width
- coverage by activity class if possible

This last one is a nice optional touch.

### Definition of done for Phase 8
You have a forecast with uncertainty, not just a point estimate.

---

# Phase 9 — Final visuals and narrative

## 9.1. Must-have figures
Create:
- activity class distribution
- example telemetry traces
- actual versus predicted heart rate on test set
- residual distribution
- confusion matrix
- interval coverage plot
- one table summarizing final metrics

## 9.2. Must-have written conclusions
Answer these explicitly in markdown:
- What did the data pipeline do?
- How well could recent telemetry forecast heart rate 30 seconds ahead?
- How well could recent telemetry classify activity state?
- Were intervals reasonably calibrated?
- What would be improved next?

## 9.3. Write the resume-safe summary
At the end, write 4 to 6 lines summarizing the project in plain English.

### Definition of done for Phase 9
A stranger could open the notebook and understand the whole project.

---

# 7. Dask rule for this MVP

## 7.1. Default decision
Do not use Dask in the core MVP unless the notebook is clearly slow.

## 7.2. When Dask is justified
Use Dask only if one of these becomes annoying:
- per-subject feature generation is slow
- repeated backtests across subjects or horizons are slow
- parquet loading or processing becomes clunky

## 7.3. If Dask is used, use it narrowly
Only use Dask for:
- parallel feature generation by subject
- parallel evaluation by fold or subject

Do not rewrite the whole project around Dask.

---

# 8. Suggested MVP schedule

## Day 1 morning
- setup
- load data
- audit one subject
- load all subjects
- decide activity scope

## Day 1 afternoon
- build per-second table
- compute magnitudes
- build lagged features
- create targets
- save processed table

## Day 2 morning
- create subject split
- run baselines
- run one stronger regressor
- run one stronger classifier

## Day 2 afternoon
- add conformal intervals
- make final figures
- write conclusions
- write project summary

---

# 9. Common failure points to avoid

## 9.1. Leakage
Do not:
- randomly split rows
- compute rolling features across subject boundaries
- let future information leak into features

## 9.2. Scope creep
Do not:
- add deep nets
- add many models
- add deployment in the MVP

## 9.3. Messy labels
Do not force every rare activity into the MVP.

## 9.4. Weak storytelling
Do not present this as "I trained a classifier on PAMAP2."

Present it as:
- "I built a wearable telemetry pipeline that supported forecasting, state classification, and calibrated uncertainty from recent sensor history."

---

# 10. What success looks like

This MVP is successful if, by the end, it honestly supports saying:

"I built a notebook-based wearable telemetry pipeline on PAMAP2 that cleaned raw sensor streams, built time-windowed features, forecasted heart rate 30 seconds ahead, classified activity state, and generated calibrated prediction intervals using split conformal evaluation under a subject-held-out split."