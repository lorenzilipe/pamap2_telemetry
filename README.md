# PAMAP2 telemetry MVP

This repository is a lean, notebook-first machine learning project on wearable telemetry.
It focuses on one practical question: how much can recent sensor history tell us about near-future heart rate and current activity, and how reliable are those predictions across people?

## Problem

Wearable data science often looks strong in random splits and weak in real use. This project aims to avoid that trap by building a compact pipeline and evaluating it under subject-grouped splits.

Why this matters:
- Heart-rate forecasting and activity recognition are common wearable tasks
- Cross-subject generalization is usually the hardest part
- Uncertainty estimates are needed if outputs are used for decisions

## Approach

### Data pipeline

The pipeline starts from PAMAP2 Protocol files and builds a 1-second telemetry table, then creates subject-specific lagged and rolling features.

Key data design choices:
- Compact feature set (sensor magnitudes plus small targeted upgrades)
- Subject-specific temporal transforms to avoid leakage
- One shared upstream feature build, then a task-specific (classification vs. regression) row filters

Task-specific tables:
- Regression-ready: selected features + selected regression target
- Classification-ready: selected features + activity target only

### Three tasks

1. Regression: forecast heart rate 30 seconds ahead (preferred target is next-30-second mean)
2. Classification: classify current activity state
3. Uncertainty: add split conformal prediction intervals to regression outputs

### Model choices

The project intentionally stays simple:
- Regression candidates: persistence baseline, linear regression, hist gradient boosting
- Classification candidates: logistic regression, random forest
- Uncertainty: split conformal prediction with global baseline and conditioned variant comparison

## Evaluation

Evaluation is grouped by subject using leave-one-subject-out cross-validation.

What is checked:
- Regression: MAE, RMSE, R2
- Classification: accuracy, macro F1, per-class and by-activity breakdowns
- Uncertainty: empirical coverage, interval width, failure patterns by subject and activity
- Confidence diagnostics: reliability bins, ECE, multiclass Brier score, abstention tradeoff
- Online-ish sensitivity: fixed 5-second heart-rate availability delay (`onlineish_hr_delay_5s`)

## Results

Current selected-model summary:

| Area | Best result |
|---|---|
| Regression | HistGradientBoosting, MAE 3.813, RMSE 5.579, R2 0.941 |
| Classification | RandomForest, macro F1 0.744, accuracy 0.777 |
| Uncertainty | global conformal coverage 0.918 at target 0.900, mean width 18.92 |
| Calibration | ECE(10) 0.060, multiclass Brier 0.329 |
| Online-ish stress check | 5-second HR delay increased regression MAE by +0.734 bpm; classification changed only slightly in this run |

Where it works well:
- Stable grouped cross-subject evaluation
- Useful short-horizon heart-rate forecasts
- Practical confidence and interval diagnostics

Where it struggles:
- Harder activities (especially stair classes) show weaker uncertainty behavior
- Forecast quality depends in part on timely heart-rate availability
- This is still an offline workflow with simulated online elements, not a streaming deployment

## Conclusions and limitations

What this project demonstrates:
- A clean wearable telemetry pipeline from raw data to model diagnostics
- Leakage-aware feature engineering and subject-grouped evaluation
- Honest uncertainty reporting instead of point predictions only

What it does not demonstrate:
- Production-level deployment or fully online adaptation
- Resilience to packet loss, jitter, drift, or real-time compute limits

Most realistic next improvements:
1. Add a small set of richer online-ish stress scenarios (delay + packet loss)
2. Improve activity-conditioned interval stability on hard classes
3. Tighten calibration for confidence-based classification decisions

## Final workflow

Run from repository root:

1. Compact ablation and preferred setup selection

```bash
python scripts/compact_ablation_study.py
```

2. Grouped evaluation, uncertainty, and confidence diagnostics

```bash
python scripts/grouped_evaluation.py
```

3. Refresh selected model records

```bash
python scripts/write_experiment_records.py
```

## Repo layout

- `notebooks/01_ingest_and_audit.ipynb`: raw ingest and dataset audit
- `notebooks/02_feature_pipeline.ipynb`: feature pipeline and compact ablation
- `notebooks/03_modeling_and_uncertainty.ipynb`: grouped modeling and diagnostics
- `src/pamap2_telemetry/`: reusable pipeline, training, evaluation, uncertainty, and record helpers
- `docs/`: scope, dataset decisions, contracts, and decision history
