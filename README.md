# PAMAP2 Telemetry MVP

Notebook-first, lean ML project on PAMAP2 wearable data.

This repository shows one shared telemetry pipeline that supports three tasks:
1. regression: forecast heart rate from recent wearable history,
2. classification: classify current activity state,
3. uncertainty: add split conformal prediction intervals to regression outputs.

Downstream modeling now uses task-specific tables:
- regression-ready table: rows valid for selected regression features + selected regression target,
- classification-ready table: rows valid for selected classification features + activity target only.

The project is intentionally simple by design:
- local only,
- Python only,
- sklearn + pandas stack,
- notebook narrative for EDA and interpretation,
- reusable code in a small `src/` package.

## Why this shape

The goal is a small production-style ML structure without overengineering:
- keep exploratory storytelling in notebooks,
- move reusable pipeline/model logic into code,
- keep transforms consistent between training and inference,
- make assumptions explicit through data contracts and experiment records.

## MVP tasks

- Regression target: `hr_target_next30s_mean` (mean heart rate over next 30 seconds).
- Classification target: `activity_target` (current activity label).
- Uncertainty method: grouped split conformal intervals on selected regression model.

## Project layout

- `src/pamap2_telemetry/`
  - `ingest.py`: protocol ingest and 1-second table preparation
  - `features.py`: lag/rolling feature + target builders
  - `splits.py`: grouped split helpers
  - `train.py`: model specs and fit/predict helpers
  - `evaluate.py`: grouped CV evaluation and diagnostics
  - `uncertainty.py`: conformal and failure analysis helpers
  - `ablation.py`: compact ablation orchestration
  - `experiment_records.py`: lightweight model record writer
- `notebooks/`
  - `01_ingest_and_audit.ipynb` (EDA + audit)
  - `02_feature_pipeline.ipynb` (thin orchestration for compact ablation)
  - `03_modeling_and_uncertainty.ipynb` (thin orchestration + reporting)
- `scripts/`
  - `compact_ablation_study.py` (thin wrapper to `src`)
  - `grouped_evaluation.py` (thin wrapper to `src`, reads preferred target from ablation artifact)
  - `write_experiment_records.py` (writes selected model records)
- `docs/`
  - scope, dataset, decision log, build recipe, and explicit data contracts

## Data contracts and model records

- Stage contracts: `docs/data_contracts.md`
- Machine-readable schemas:
  - `docs/schemas/raw_input_schema.json`
  - `docs/schemas/interim_telemetry_schema.json`
  - `docs/schemas/model_table_schema.json`
  - `docs/schemas/prediction_output_schema.json`
- Experiment records:
  - `docs/model_records/regression_selected_model_record.json`
  - `docs/model_records/classification_selected_model_record.json`
  - artifact copies are also written under `artifacts/models/metadata/`

## Reproducible rerun flow

From repository root:

1. Build compact table + ablation summaries
```bash
python scripts/compact_ablation_study.py
```

2. Run grouped evaluation + uncertainty and confidence diagnostics
```bash
python scripts/grouped_evaluation.py
```

3. Refresh selected-model records
```bash
python scripts/write_experiment_records.py
```

Important rerun rule:
- Step 2 reads `artifacts/metrics/grouped_cv_preferred_setup_summary.csv` and uses `preferred_target_col` as the regression target.
- If that artifact is missing or inconsistent, Step 2 fails explicitly instead of silently falling back to a different target.
- Step 1 writes task-specific processed tables under `data/processed/`:
  - `pamap2_model_table_regression.parquet`
  - `pamap2_model_table_classification.parquet`

Notebook users can run the same flow from:
- `notebooks/02_feature_pipeline.ipynb`
- `notebooks/03_modeling_and_uncertainty.ipynb`

## Final methods (selected)

- Regression model: `hist_gradient_boosting` (with in-pipeline median imputation).
- Classification model: `random_forest` (median imputation + random forest pipeline).
- Split strategy: leave-one-subject-out grouped CV.
- Preferred conformal variant: global split conformal.
- Online-ish stress mode: `onlineish_hr_delay_5s` (fixed 5-second HR availability delay before HR-derived feature generation).

## Key grouped results

From current metric artifacts:

- Preferred setup:
  - feature set: `upgraded`
  - target: `hr_target_next30s_mean`
  - fill strategy: `current_ffill`
- Regression (`hist_gradient_boosting`):
  - mean MAE: `3.813`
  - mean RMSE: `5.579`
  - mean R2: `0.941`
  - MAE gain vs persistence baseline: `0.318`
- Classification (`random_forest`):
  - mean macro F1: `0.744`
  - mean accuracy: `0.777`
- Uncertainty (global conformal):
  - row-level empirical coverage: `0.918` (target `0.900`)
  - mean interval width: `18.920`
- Confidence diagnostics:
  - ECE(10): `0.073`
  - multiclass Brier score: `0.306`

## Online-ish delayed-HR check

One lightweight online-ish mode is now run in compact ablation to test dependence on offline convenience assumptions:
- mode: `onlineish_hr_delay_5s`
- simulated condition: heart-rate input is available only with a fixed 5-second lag
- causal rule: HR-derived features use only delayed HR values, while non-HR telemetry stays current
- target rule: regression targets remain true future HR values

Compact comparison (`artifacts/metrics/grouped_cv_onlineish_comparison_summary.csv`):

| evaluation_mode | hr_delay_seconds | regression_mean_mae | regression_mean_rmse | regression_mean_r2 | classification_mean_macro_f1 | classification_mean_accuracy |
|---|---:|---:|---:|---:|---:|---:|
| offline_standard | 0 | 3.813 | 5.579 | 0.941 | 0.744 | 0.777 |
| onlineish_hr_delay_5s | 5 | 4.546 | 6.473 | 0.920 | 0.745 | 0.785 |

What changed:
- regression MAE increased by `+0.734` bpm
- regression RMSE increased by `+0.894` bpm
- regression R2 decreased by `-0.021`
- classification macro F1 and accuracy changes stayed small in this run (`+0.002` and `+0.009`)

Interpretation:
- Forecast performance remains useful, but a meaningful part of regression accuracy depends on immediate HR availability.
- Classification appears less sensitive to this delayed-HR condition in the current feature/model setup.
- The largest online-ish risk signal is in regression error growth, not in class discrimination.
- Activity-level regression sensitivity is uneven: descending stairs, running, and nordic walking show the largest MAE increases in `artifacts/metrics/grouped_cv_onlineish_regression_activity_delta.csv`.

## Where it performs well

- Stable grouped CV workflow with strict subject isolation.
- Compact feature upgrades improved regression performance without feature explosion.
- Classification probabilities are useful enough for confidence-aware filtering.
- Conformal layer gives practical interval diagnostics by activity and subject.

## Where it struggles

- Hard activities still have weak interval behavior (for example, stair classes).
- Forecasting remains partly persistence-driven.
- Calibration quality is usable but not perfect under harder regimes.
- This is not a deployment-grade system (no drift handling, no streaming constraints).

## Limitations Of The Online-ish Check

- This is a lightweight simulation, not a streaming deployment test.
- Only one delay scenario is simulated (fixed 5-second HR lag).
- It does not model packet loss, variable latency jitter, device clock drift, or inference-time compute delays.
- It does not include online model updates or drift adaptation.
- It shows sensitivity to stale HR inputs, but does not prove production readiness.

## Why simplicity is explicit

This repository chooses transparent, defensible methods over complexity:
- small model set,
- compact feature engineering,
- grouped CV for realism,
- uncertainty as practical diagnostics,
- clear artifact trail.

That keeps the project easy to explain in interviews while still showing real ML pipeline discipline.
