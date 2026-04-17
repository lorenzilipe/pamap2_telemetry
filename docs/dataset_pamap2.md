# PAMAP2 dataset notes

This document records how PAMAP2 is interpreted and used in this repository.
For formal stage contracts, see `docs/data_contracts.md`.

## dataset use in this project

PAMAP2 is used to support three tasks from one shared telemetry pipeline:
1. near-future heart-rate regression,
2. current activity classification,
3. uncertainty intervals for regression.

The project intentionally stays compact and interpretable.

## raw source and key fields

Primary source:
- `data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset/Protocol/subject*.dat`

Key raw fields used:
- `timestamp_s`
- `activity_id`
- `heart_rate_bpm`
- hand/chest/ankle accelerometer axes (`*_acc_16g_x|y|z`)
- hand/chest/ankle gyroscope axes (`*_gyro_x|y|z`)

Subject ID is parsed from filename.

## activity scope

Kept activity IDs:
- `1` lying
- `2` sitting
- `3` standing
- `4` walking
- `5` running
- `6` cycling
- `7` nordic_walking
- `12` ascending_stairs
- `13` descending_stairs
- `16` vacuum_cleaning
- `17` ironing

Rows with `activity_id = 0` are treated as transient/unlabeled and excluded from supervised modeling.

## missingness and heart-rate policy

Observed from audit:
- raw HR missingness is high at 100 Hz and expected for PAMAP2 sampling rates

Current policy:
- aggregate to 1-second rows first
- apply subject-local forward fill (`current_ffill`) as default
- keep evaluated alternatives for sensitivity:
  - `limited_ffill_5s`
  - `strict_observed_only`

## 1-second telemetry table

Interim output:
- `data/interim/pamap2_per_second.parquet`

Core columns:
- identifiers: `subject_id`, `timestamp_s`, `activity_id`, `activity_label`
- HR columns: `heart_rate_bpm`, `heart_rate_observed_flag`
- compact magnitudes:
  - `hand_acc_16g_mag`, `chest_acc_16g_mag`, `ankle_acc_16g_mag`
  - `hand_gyro_mag`, `chest_gyro_mag`, `ankle_gyro_mag`
- compact axis summaries used in upgraded feature set:
  - `hand_acc_axis_absmean`, `chest_acc_axis_absmean`

## feature and target definitions

Baseline feature family:
- current value, lag-1, lag-5,
- rolling mean/std over 5 and 10 seconds,
- short-term delta from rolling mean

Targeted upgraded feature family adds:
- HR shape stats (min/max/median/quantiles over short windows)
- HR baseline-relative features
- transition-sensitive motion features
- tiny axis summary subset

Target columns built upstream:
- `activity_target`
- `hr_target_30s`
- `hr_target_15s`
- `hr_target_next30s_mean`

Preferred regression target:
- `hr_target_next30s_mean`

## task-specific processed tables

Shared upstream feature generation feeds two downstream tables:

- regression table:
  - `data/processed/pamap2_model_table_regression.parquet`
  - row requires selected features + selected regression target

- classification table:
  - `data/processed/pamap2_model_table_classification.parquet`
  - row requires selected features + `activity_target`

Reason for split:
- classification should not lose valid rows because of future-target availability needed only by regression.

## evaluation design

Primary evaluation:
- leave-one-subject-out grouped CV

Uncertainty:
- split conformal on grouped folds
- global variant kept as preferred in current run
- activity-conditioned variant retained as comparison output

Online-ish sensitivity check:
- mode: `onlineish_hr_delay_5s`
- simulates 5-second HR availability delay before HR-derived feature generation
- used to quantify reliance on immediate HR access

## key outputs

Core metrics and diagnostics are written under `artifacts/metrics/`, including:
- grouped model summaries and fold metrics
- conformal diagnostics
- calibration and abstention diagnostics
- online-ish comparison outputs

Selected model records are written to:
- `docs/model_records/regression_selected_model_record.json`
- `docs/model_records/classification_selected_model_record.json`

## leakage controls

Main safeguards used throughout the pipeline:
- sort by `subject_id` and `timestamp_s` before temporal transforms
- compute lag/rolling/shift operations within subject only
- keep subject isolation in evaluation folds
- keep feature construction past-only with future values used only for targets
