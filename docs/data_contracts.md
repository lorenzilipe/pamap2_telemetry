# data contracts and stage schemas

This document defines stage contracts for the final MVP workflow.

Scope is intentionally lean:
- one shared telemetry pipeline
- one preferred regression target
- one activity classification target
- one uncertainty layer

## stage 1: raw protocol input

Source files:
- `data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset/Protocol/subject*.dat`

Required raw fields:
- `timestamp_s`
- `activity_id`
- `heart_rate_bpm`
- hand/chest/ankle accelerometer axes (`*_acc_16g_x|y|z`)
- hand/chest/ankle gyroscope axes (`*_gyro_x|y|z`)

Validity rules:
- subject ID parsed from filename
- rows sorted by `timestamp_s` before temporal operations
- `activity_id = 0` excluded from supervised modeling
- no cross-subject filling

## stage 2: 1-second telemetry table

Output file:
- `data/interim/pamap2_per_second.parquet`

Required columns:
- identifiers: `subject_id`, `session`, `timestamp_s`, `activity_id`, `activity_label`
- HR columns: `heart_rate_bpm`, `heart_rate_observed_flag`
- magnitudes:
  - `hand_acc_16g_mag`, `chest_acc_16g_mag`, `ankle_acc_16g_mag`
  - `hand_gyro_mag`, `chest_gyro_mag`, `ankle_gyro_mag`
- optional compact axis summaries:
  - `hand_acc_axis_absmean`, `chest_acc_axis_absmean`

Validity rules:
- one row per `(subject_id, timestamp_s)`
- allowed activity IDs: `1,2,3,4,5,6,7,12,13,16,17`
- magnitude formula: `sqrt(x^2 + y^2 + z^2)`

HR handling rules:
- subject-local fill only
- supported fill strategy labels:
  - `current_ffill`
  - `limited_ffill_5s`
  - `strict_observed_only`

## stage 3: shared upstream features and targets

This stage is built once per preferred setup and reused by both tasks.

Shared upstream columns:
- identifiers: `subject_id`, `session`, `timestamp_s`, `activity_id`, `activity_label`
- metadata: `heart_rate_observed_flag`, `heart_rate_fill_strategy`
- targets: `activity_target`, `hr_target_30s`, `hr_target_15s`, `hr_target_next30s_mean`
- selected feature columns from baseline/upgraded set

Rules:
- all temporal transforms are subject-local
- data is sorted by `subject_id`, then `timestamp_s`
- classification row eligibility is not tied to regression target availability

## stage 4: task-specific processed tables

### stage 4a: regression-ready table

Output:
- `data/processed/pamap2_model_table_regression.parquet`

Row eligibility:
- non-null selected feature columns
- non-null selected regression target (`preferred_target_col`)

Required columns:
- `subject_id`, `timestamp_s`, `activity_id`, `activity_label`
- `activity_target`
- selected regression target column

### stage 4b: classification-ready table

Output:
- `data/processed/pamap2_model_table_classification.parquet`

Row eligibility:
- non-null selected feature columns
- non-null `activity_target`

Required columns:
- `subject_id`, `timestamp_s`, `activity_id`, `activity_label`
- `activity_target`

Design rule:
- regression and classification share upstream features but diverge at downstream row filtering.

## stage 5: evaluation and prediction outputs

Primary metric outputs:
- `artifacts/metrics/grouped_cv_regression_fold_metrics.csv`
- `artifacts/metrics/grouped_cv_classification_fold_metrics.csv`
- `artifacts/metrics/grouped_cv_selected_model_summary.csv`
- `artifacts/metrics/grouped_cv_regression_summary.csv`
- `artifacts/metrics/grouped_cv_classification_summary.csv`

Uncertainty outputs:
- `artifacts/metrics/grouped_cv_conformal_summary.csv`
- `artifacts/metrics/grouped_cv_conformal_summary_all_variants.csv`
- `artifacts/metrics/grouped_cv_conformal_predictions.csv`

Confidence outputs:
- `artifacts/metrics/grouped_cv_classification_calibration_summary.csv`
- `artifacts/metrics/grouped_cv_classification_reliability_by_bin.csv`
- `artifacts/metrics/grouped_cv_classification_abstention_summary.csv`

Online-ish sensitivity outputs:
- `artifacts/metrics/grouped_cv_onlineish_comparison_summary.csv`
- `artifacts/metrics/grouped_cv_onlineish_regression_activity_delta.csv`

Prediction-level requirements:
- regression predictions: `y_true`, `y_pred`, `subject_id`, `timestamp_s`
- classification predictions: `y_true`, `y_pred`, confidence, per-class probabilities
- conformal predictions: `lower`, `upper`, `covered`, `interval_width`

## preferred target contract for grouped reruns

- grouped evaluation reads `artifacts/metrics/grouped_cv_preferred_setup_summary.csv`
- `preferred_target_col` from that file is required and used as regression target
- if missing or inconsistent, grouped evaluation must fail explicitly

## model record contract

Canonical tracked records:
- `docs/model_records/regression_selected_model_record.json`
- `docs/model_records/classification_selected_model_record.json`

Runtime copies may also be written under `artifacts/models/metadata/`.

Required content:
- regression record: selected regression metrics plus conformal coverage and interval width
- classification record: selected classification metrics plus calibration/confidence diagnostics
