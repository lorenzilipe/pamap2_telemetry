# Data Contracts And Stage Schemas

This document defines explicit contracts for each pipeline stage in the MVP.

Scope is intentionally lean:
- one shared telemetry pipeline,
- one regression target,
- one classification target,
- one uncertainty layer.

## Stage 1: Raw PAMAP2 Input

Source:
- `data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset/Protocol/subject*.dat`

Required columns:
- `timestamp_s`: elapsed time in seconds.
- `activity_id`: integer activity code from metadata mapping.
- `heart_rate_bpm`: beats per minute.
- `hand_acc_16g_x`, `hand_acc_16g_y`, `hand_acc_16g_z`: hand accelerometer axes.
- `hand_gyro_x`, `hand_gyro_y`, `hand_gyro_z`: hand gyroscope axes.
- `chest_acc_16g_x`, `chest_acc_16g_y`, `chest_acc_16g_z`: chest accelerometer axes.
- `chest_gyro_x`, `chest_gyro_y`, `chest_gyro_z`: chest gyroscope axes.
- `ankle_acc_16g_x`, `ankle_acc_16g_y`, `ankle_acc_16g_z`: ankle accelerometer axes.
- `ankle_gyro_x`, `ankle_gyro_y`, `ankle_gyro_z`: ankle gyroscope axes.

Assumptions and validity rules:
- Subject ID is parsed from filename (`subject101.dat` -> `subject_id=101`).
- Rows are sorted by `timestamp_s` before temporal operations.
- `activity_id=0` is treated as transient/unlabeled and excluded from supervised modeling.
- Protocol files are the canonical MVP source to avoid Optional-session leakage.

Missingness handling:
- Raw HR missingness is expected at high frequency.
- No cross-subject filling is allowed.
- Missingness audits are written to `artifacts/metrics/phase1_*`.

## Stage 2: 1-Second Telemetry Table

Output:
- `data/interim/pamap2_per_second.parquet`

Required columns:
- `subject_id`
- `session`
- `timestamp_s`
- `activity_id`
- `activity_label`
- `heart_rate_bpm`
- `heart_rate_observed_flag`
- `hand_acc_16g_mag`, `chest_acc_16g_mag`, `ankle_acc_16g_mag`
- `hand_gyro_mag`, `chest_gyro_mag`, `ankle_gyro_mag`
- optional compact raw-axis summaries used by upgraded feature set:
  - `hand_acc_axis_absmean`, `chest_acc_axis_absmean`

Assumptions and validity rules:
- One row per `(subject_id, timestamp_s)`.
- Only kept activity IDs are present (`1,2,3,4,5,6,7,12,13,16,17`).
- Magnitude features are computed as `sqrt(x^2 + y^2 + z^2)`.

Missingness handling:
- HR fill strategy is subject-local only.
- Fill strategy names:
  - `current_ffill`
  - `limited_ffill_5s`
  - `strict_observed_only`

## Stage 3: Final Model Table

Output:
- `data/processed/pamap2_model_table.parquet`

Required identifier/target columns:
- `subject_id`, `timestamp_s`, `activity_id`, `activity_label`
- `activity_target`
- `hr_target_30s`
- `hr_target_15s`
- `hr_target_next30s_mean`
- `heart_rate_fill_strategy`

Required feature groups:
- core signals: `heart_rate_bpm`, accel magnitudes, gyro magnitudes
- baseline temporal features:
  - lags (`_lag_1`, `_lag_5`)
  - rolling means/std (`_rollmean_5`, `_rollstd_5`, `_rollmean_10`, `_rollstd_10`)
  - delta-from-recent-baseline (`_delta_from_rollmean_5`)
- upgraded compact features:
  - HR shape and baseline-relative features
  - transition-sensitive motion summaries
  - tiny raw-axis summary subset

Assumptions and validity rules:
- Temporal features and targets are computed within subject only.
- Table is sorted by `subject_id`, then `timestamp_s` before lags/rolling/shifts.
- Any row used for model fitting must have non-null selected features and selected target.

Missingness handling:
- Missingness is handled in two layers:
  - row filtering for required training columns,
  - in-pipeline median imputation in sklearn Pipelines for model robustness.

## Stage 4: Model Output Contracts

Primary output files:
- grouped regression fold metrics:
  - `artifacts/metrics/grouped_cv_regression_fold_metrics.csv`
- grouped classification fold metrics:
  - `artifacts/metrics/grouped_cv_classification_fold_metrics.csv`
- selected model summary:
  - `artifacts/metrics/grouped_cv_selected_model_summary.csv`
- conformal summaries:
  - `artifacts/metrics/grouped_cv_conformal_summary.csv`
  - `artifacts/metrics/grouped_cv_conformal_summary_all_variants.csv`
- prediction-level outputs:
  - `artifacts/metrics/grouped_cv_regression_predictions_all_models.csv`
  - `artifacts/metrics/grouped_cv_classification_predictions_all_models.csv`
  - `artifacts/metrics/grouped_cv_conformal_predictions.csv`

Prediction output requirements:
- regression predictions include `y_true`, `y_pred`, `subject_id`, `timestamp_s`.
- classification predictions include `y_true`, `y_pred`, confidence scores, and per-class probabilities.
- conformal predictions include `lower`, `upper`, `covered`, and `interval_width`.

Experiment record files:
- `docs/model_records/regression_selected_model_record.json`
- `docs/model_records/classification_selected_model_record.json`

Runtime copies are also written to `artifacts/models/metadata/` for local runs.

These records provide task, target, split strategy, metrics summary, and uncertainty notes for selected models.
