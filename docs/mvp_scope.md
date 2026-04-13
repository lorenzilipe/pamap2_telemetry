# MVP scope

## Current project version

Version: v1 ultra-lean notebook MVP

## Core objective

Build the strongest possible weekend-scale wearable telemetry project without overbuilding. The MVP must stay narrow enough to finish quickly while still being rigorous enough to discuss confidently in interviews.

## Primary tasks

### Task 1: regression
Forecast heart rate 30 seconds ahead.

### Task 2: classification
Classify current activity state from recent wearable telemetry.

### Task 3: uncertainty
Generate lightweight prediction intervals for the regression task using split conformal prediction.

## Canonical target definitions

### Regression target
Primary target objective:
- `heart_rate` at `t + 30 seconds`

Required sensitivity targets:
- average heart rate over the next 30 seconds
- one shorter direct horizon (15 seconds in current implementation)

Current preferred grouped-evaluation target (2026-04-12 update):
- average heart rate over the next 30 seconds (`hr_target_next30s_mean`)

Why:
- it materially improved grouped held-out MAE versus direct `t + 30` while keeping the task short-horizon and interview-defensible.

### Classification target
Default target:
- current activity label at time `t`

The MVP does not need to predict future activity transitions.

### Uncertainty target
Prediction intervals around the heart-rate forecast.

## Canonical data representation

Use a 1-second telemetry table.

Each row should represent what is known for one subject at one second. The row should contain:

- subject ID
- timestamp
- current activity label
- heart rate
- compact sensor summary features
- lagged and rolling features created only from recent past data

## Feature philosophy

The MVP should prioritize compact, interpretable telemetry features instead of large raw-signal expansions.

Start with:

- accelerometer magnitude by body location
- gyroscope magnitude by body location
- optional temperature by body location
- heart rate
- lagged values
- rolling means
- rolling standard deviations
- short-term change features

Targeted upgrades allowed in v1 (small and explicit only):
- rolling min/max/median and limited quantiles for heart rate
- transition-sensitive motion summaries (recent change and variance burst)
- recent-baseline-relative heart-rate features
- at most a tiny raw-axis summary subset when justified

## Activity scope

Do not force all activities into v1.

During the audit phase:
- inspect class counts,
- keep the major activity classes,
- drop rare, low-support, or messy classes if needed.

The goal is a clean, defensible MVP, not maximal label coverage.

## Evaluation design

### Split rule
Use subject-grouped evaluation with leave-one-subject-out cross-validation.

Default logic:
- each fold holds out one full subject for evaluation
- no subject appears in both train and fold-test data
- all model comparisons use the same grouped folds

If runtime becomes too heavy, use grouped K-fold while preserving full subject isolation.

### Metrics

#### Regression
- MAE
- RMSE
- optional R^2

#### Uncertainty
- empirical coverage
- average interval width

#### Classification
- accuracy
- macro F1
- confusion matrix

## Required deliverables

By the end of the MVP, the repo should contain:

- raw data in `data/raw/`
- 1-second telemetry table in `data/interim/`
- final modeling table in `data/processed/`
- lean reusable package in `src/pamap2_telemetry/` for ingest/features/splits/train/evaluate/uncertainty logic
- three notebooks for ingestion, feature engineering, and modeling
- saved figures
- saved metrics tables
- lightweight experiment record files for selected models
- explicit stage contracts for data and prediction outputs
- updated docs

## Non-goals for v1

These are not required for the MVP:

- API
- dashboard
- cloud setup
- Docker
- CI/CD
- Dask unless truly justified
- deep learning
- hyperparameter sweeps
- a large model zoo

## Modeling defaults

### Regression model defaults
Start with:
- persistence baseline using current heart rate
- one linear model (`LinearRegression`)
- one tree-based model (`HistGradientBoostingRegressor` preferred)

### Classification model defaults
Start with:
- one linear classifier baseline (`LogisticRegression`)
- one tree-based model (`RandomForestClassifier`)

### Uncertainty default
Use split conformal prediction on top of the final regression model.

## Scope guardrails

If a proposed change would add significant complexity, first ask:

1. Does this improve the core signal of forecasting, telemetry, or data pipelines?
2. Is this necessary for the MVP?
3. Can this be explained clearly in an interview?
4. Will this delay completion more than it improves project value?

If the answer is mostly no, do not add it.

## What the final project should honestly support saying

"I built a notebook-based wearable telemetry pipeline on PAMAP2 that cleaned raw sensor streams, built time-windowed features, forecasted heart rate 30 seconds ahead, classified activity state, and generated calibrated prediction intervals using split conformal evaluation under subject-grouped cross-validation."
