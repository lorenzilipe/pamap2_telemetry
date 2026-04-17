# PAMAP2 telemetry MVP

This repository is a lean, notebook-first machine learning project on wearable telemetry.
It focuses on one practical question: how much can recent sensor history tell us about near-future heart rate and current activity, and how reliable are those predictions across people?

## Problem

Wearable data science often looks strong in random splits and weak in real use. This project aims to avoid that trap by building a compact pipeline and evaluating it under subject-grouped splits.

Why this matters:
- heart-rate forecasting and activity recognition are common wearable tasks
- cross-subject generalization is usually the hardest part
- uncertainty estimates are needed if outputs are used for decisions

## Approach

### Data pipeline

The pipeline starts from PAMAP2 Protocol files and builds a 1-second telemetry table, then creates subject-local lagged and rolling features.

Key design choices:
- compact feature set (sensor magnitudes plus small targeted upgrades)
- subject-local temporal transforms to avoid leakage
- one shared upstream feature build, then task-specific downstream row filters

Task-specific tables:
- regression-ready: selected features + selected regression target
- classification-ready: selected features + activity target only

### Three tasks

1. regression: forecast heart rate 30 seconds ahead (preferred target is next-30-second mean)
2. classification: classify current activity state
3. uncertainty: add split conformal prediction intervals to regression outputs

### Model choices

The project intentionally stays simple:
- regression candidates: persistence baseline, linear regression, hist gradient boosting
- classification candidates: logistic regression, random forest
- uncertainty: split conformal with global baseline and conditioned variant comparison

This keeps the work explainable and interview-defensible without adding low-ROI complexity.

## Evaluation

Evaluation is grouped by subject using leave-one-subject-out cross-validation.

What is checked:
- regression: MAE, RMSE, R2
- classification: accuracy, macro F1, per-class and by-activity breakdowns
- uncertainty: empirical coverage, interval width, failure patterns by subject and activity
- confidence diagnostics: reliability bins, ECE, multiclass Brier score, abstention tradeoff
- online-ish sensitivity: fixed 5-second heart-rate availability delay (`onlineish_hr_delay_5s`)

## Results

Current selected-model summary:

| area | headline result |
|---|---|
| regression | HistGradientBoosting, MAE 3.813, RMSE 5.579, R2 0.941 |
| classification | RandomForest, macro F1 0.744, accuracy 0.777 |
| uncertainty | global conformal coverage 0.918 at target 0.900, mean width 18.92 |
| calibration | ECE(10) 0.060, multiclass Brier 0.329 |
| online-ish stress check | 5-second HR delay increased regression MAE by +0.734 bpm; classification changed only slightly in this run |

Where it works well:
- stable grouped cross-subject evaluation
- useful short-horizon heart-rate forecasts
- practical confidence and interval diagnostics

Where it struggles:
- harder activities (especially stair classes) show weaker uncertainty behavior
- forecast quality depends in part on timely heart-rate availability
- this is still an offline workflow, not a streaming deployment

## Conclusions and limitations

What this project demonstrates:
- a clean wearable telemetry pipeline from raw data to model diagnostics
- leakage-aware feature engineering and subject-grouped evaluation
- honest uncertainty reporting instead of point predictions only

What it does not demonstrate:
- production deployment or online adaptation
- resilience to packet loss, jitter, drift, or real-time compute limits

Most realistic next improvements:
1. add a small set of richer online-ish stress scenarios (delay + packet loss)
2. improve activity-conditioned interval stability on hard classes
3. tighten calibration for confidence-based classification decisions

## Canonical workflow

Run from repository root:

1. compact ablation and preferred setup selection

```bash
python scripts/compact_ablation_study.py
```

2. grouped evaluation, uncertainty, and confidence diagnostics

```bash
python scripts/grouped_evaluation.py
```

3. refresh selected model records

```bash
python scripts/write_experiment_records.py
```

## Repository layout

- `notebooks/01_ingest_and_audit.ipynb`: raw ingest and dataset audit
- `notebooks/02_feature_pipeline.ipynb`: feature pipeline and compact ablation
- `notebooks/03_modeling_and_uncertainty.ipynb`: grouped modeling and diagnostics
- `src/pamap2_telemetry/`: reusable pipeline, training, evaluation, uncertainty, and record helpers
- `docs/`: scope, dataset decisions, contracts, and decision history

## Environment

This project is local-only and Python-only.

- `requirements.txt`: pip environment
- `environment.yml`: conda environment
- `scripts/setup_mac.sh` and `scripts/setup_windows.ps1`: convenience setup scripts

---

Honest project summary:

"Built a notebook-based wearable telemetry pipeline on PAMAP2 that cleaned raw sensor streams, built time-windowed features, forecasted heart rate 30 seconds ahead, classified activity state, and generated calibrated prediction intervals under subject-grouped evaluation."
