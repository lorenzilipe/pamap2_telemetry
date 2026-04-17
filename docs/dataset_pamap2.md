# PAMAP2 dataset working notes

## Purpose of this document

This file is the project-specific source of truth for how PAMAP2 is being interpreted and used in this repository.

It should answer:
- what raw data fields matter,
- how the dataset is being cleaned,
- how timestamps and labels are handled,
- how the 1-second telemetry table is built,
- how final modeling features and targets are defined.

Explicit contract references:
- canonical stage contracts: `docs/data_contracts.md`
- machine-readable schemas: `docs/schemas/*.json`

This document starts partially templated and should be updated after Phase 1.

---

## Current use of PAMAP2 in this project

PAMAP2 is being used to support a shared wearable telemetry pipeline for three tasks:

1. heart-rate forecasting,
2. activity-state classification,
3. lightweight uncertainty estimation.

The project is not trying to model every possible sensor nuance in the dataset. It is using PAMAP2 to build a compact, interpretable MVP.

---

## Phase 1 audit checklist

During ingestion and audit, capture the following here.

### Raw file locations
- raw files location: `data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset/`
- subject file naming pattern: `Protocol/subject<id>.dat` and `Optional/subject<id>.dat` with IDs `101` to `109`
- documentation file path: `data/raw/pamap2+physical+activity+monitoring/readme.pdf` and `data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset/*.pdf`

### Core raw fields to identify
- timestamp column: `timestamp_s` (raw column index 1)
- subject ID source: extracted from filename pattern `subject<id>.dat`
- activity label column: `activity_id` (raw column index 2)
- heart-rate column: `heart_rate_bpm` (raw column index 3)
- hand accelerometer columns: `hand_acc_16g_x`, `hand_acc_16g_y`, `hand_acc_16g_z` (raw 5-7)
- chest accelerometer columns: `chest_acc_16g_x`, `chest_acc_16g_y`, `chest_acc_16g_z` (raw 22-24)
- ankle accelerometer columns: `ankle_acc_16g_x`, `ankle_acc_16g_y`, `ankle_acc_16g_z` (raw 39-41)
- hand gyroscope columns: `hand_gyro_x`, `hand_gyro_y`, `hand_gyro_z` (raw 11-13)
- chest gyroscope columns: `chest_gyro_x`, `chest_gyro_y`, `chest_gyro_z` (raw 28-30)
- ankle gyroscope columns: `ankle_gyro_x`, `ankle_gyro_y`, `ankle_gyro_z` (raw 45-47)
- temperature columns, if used: `hand_temperature_c`, `chest_temperature_c`, `ankle_temperature_c`

### Initial audit outputs to record
- rows per subject (Protocol):
	- 101: 376,417
	- 102: 447,000
	- 103: 252,833
	- 104: 329,576
	- 105: 374,783
	- 106: 361,817
	- 107: 313,599
	- 108: 408,031
	- 109: 8,477
- heart-rate missingness by subject (Protocol): around 90.86% in raw 100 Hz rows for all subjects, consistent with PAMAP2 docs because HR is sampled near 9 Hz
- selected-column missingness table (Protocol raw ingest): saved to `artifacts/metrics/phase1_protocol_column_missingness.csv`
- activity counts overall (Protocol rows):
	- activity `0` (transient): 929,661 rows
	- highest labeled activities: walking (238,761), ironing (238,690), lying (192,523), standing (189,931)
	- detailed table saved to `artifacts/metrics/phase1_protocol_activity_counts_overall.csv`
- activity counts by subject: saved to `artifacts/metrics/phase1_protocol_activity_counts_by_subject.csv`
- observed heart-rate summary stats (Protocol raw ingest): saved to `artifacts/metrics/phase1_protocol_hr_summary_stats.csv`
- invalid or unlabeled row counts (Protocol):
	- from 24.61% (subject 109) to 41.09% (subject 102)
	- detailed table saved to `artifacts/metrics/phase1_protocol_invalid_or_unlabeled_by_subject.csv`
- quick audit time-series plots (subject 104):
	- heart rate: `artifacts/figures/phase1_protocol_hr_timeseries_subject104.png`
	- hand accelerometer magnitude: `artifacts/figures/phase1_protocol_hand_acc_timeseries_subject104.png`
- obvious anomalies or caveats:
	- `activity_id = 0` occupies a large share of rows and must be dropped for supervised modeling targets
	- subject 109 has much less protocol coverage than all other subjects
	- Optional sessions overlap Protocol subjects (101, 105, 106, 108, 109) and must remain separate in evaluation to avoid leakage
	- a neat raw-data metadata layer was added at `data/raw/pamap2+physical+activity+monitoring/metadata/` with schema, labels, subject info, manifests, and documentation digest

---

## Working cleaning rules

These are the current intended cleaning rules. Update them if implementation changes.

### Subject handling
- keep subject identity as a first-class field
- all rolling, lagging, shifting, and filling must happen within subject only

### Activity handling
- keep only the major activity classes chosen after the audit
- drop rows with invalid or unlabeled activity if they are not useful for the MVP
- document the final kept activity set below

Final kept activity set:
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

Reason:
- Selected from Protocol activity support using a transparent threshold:
	- at least 6 subjects with non-zero coverage
	- at least 90,000 raw rows in total
- This keeps major, interview-defensible classes and removes low-support activity labels for v1.

### Heart-rate handling
Current default plan:
- inspect missingness carefully during audit
- after resampling to 1-second rows, forward-fill within subject
- optionally backfill only short gaps if needed
- do not use complex imputation in v1

If a different rule is adopted, document it here.

Implemented in Phase 1:
- Kept raw HR values during 100 Hz ingest for auditing.
- Aggregated to 1-second rows with mean HR per second.
- Applied forward-fill within subject after 1-second aggregation.

Implemented sensitivity study (2026-04-12):
- Compared three subject-local HR fill strategies on the same grouped CV setup:
	- `current_ffill` (unbounded forward fill)
	- `limited_ffill_5s` (forward fill with 5-second cap)
	- `strict_observed_only` (no gap fill)
- Main result:
	- `current_ffill` and `limited_ffill_5s` produced identical grouped MAE on the upgraded direct-30s setup (`6.5691`)
	- `strict_observed_only` was slightly worse (`6.5742`) and used fewer rows
- Adoption for preferred setup:
	- keep `current_ffill` as the default due stable performance and simpler explanation.

### Time handling
- sort within subject by timestamp before any temporal operation
- resample to 1-second intervals for the main telemetry table
- use only past information to create features
- use future information only to define targets

Implemented in Phase 1:
- Sorted every subject file by `timestamp_s` before aggregation.
- Binned to 1-second resolution with `floor(timestamp_s)` and aggregated within each subject only.
- Saved interim output to `data/interim/pamap2_per_second.parquet` with 18,939 rows after kept-activity filtering.

Phase 2 strict verification (2026-04-08):
- Validation file: `artifacts/metrics/phase2_strict_validation_checks.csv`
- Additional HR check file: `artifacts/metrics/phase2_interim_hr_missingness_by_subject.csv`
- All strict checks passed:
	- expected interim schema matches exactly
	- one row per subject-second (no duplicates)
	- timestamps are sorted within each subject
	- activity IDs are limited to `1,2,3,4,5,6,7,12,13,16,17`
	- no missing `heart_rate_bpm` after subject-local forward-fill

---

## Implemented Phase 3 compact telemetry feature set

The MVP keeps the compact baseline and adds a small, targeted upgrade set.

### Core numeric signals
- `heart_rate_bpm`
- `hand_acc_16g_mag`, `chest_acc_16g_mag`, `ankle_acc_16g_mag`
- `hand_gyro_mag`, `chest_gyro_mag`, `ankle_gyro_mag`

Compact baseline (still used):
- 56 total baseline features (7 core signals + 49 lag/rolling/delta features)
- no temperature features in v1 to keep scope lean

### Baseline derived features
For each core signal, the baseline set includes:
- current value
- lag 1 second
- lag 5 seconds
- rolling mean over last 5 seconds
- rolling std over last 5 seconds
- rolling mean over last 10 seconds
- rolling std over last 10 seconds
- short-term change from rolling mean

Column naming pattern:
- `<signal>_lag_1`
- `<signal>_lag_5`
- `<signal>_rollmean_5`
- `<signal>_rollstd_5`
- `<signal>_rollmean_10`
- `<signal>_rollstd_10`
- `<signal>_delta_from_rollmean_5`

### Targeted upgraded features (2026-04-12)
Added 19 features only, focused on high-value telemetry signals:
- heart-rate shape features:
	- `heart_rate_bpm_rollmin_10`
	- `heart_rate_bpm_rollmax_10`
	- `heart_rate_bpm_rollmedian_10`
	- `heart_rate_bpm_rollq25_10`
	- `heart_rate_bpm_rollq75_10`
- heart-rate relative/baseline features:
	- `heart_rate_bpm_recent_change_5`
	- `heart_rate_bpm_rollmean_60`
	- `heart_rate_bpm_vs_rollmean_60`
- transition-sensitive motion features:
	- `motion_intensity_mean`
	- `motion_abs_change_1`
	- `motion_abs_change_5`
	- `motion_rollstd_5`
	- `motion_rollstd_20`
	- `motion_variance_burst_ratio`
	- `acc_location_dispersion`
- tiny raw-axis summary subset:
	- `hand_acc_axis_absmean`
	- `chest_acc_axis_absmean`
	- `hand_acc_axis_absmean_rollmean_5`
	- `chest_acc_axis_absmean_rollmean_5`

Upgraded set size:
- 75 total features
- explicit gain over baseline in grouped direct-30s setup:
	- MAE improved from `6.9374` to `6.5691` (`+0.3682` better)

Leakage guard used:
- All lag, rolling, and target shift operations are computed with subject-local groupby logic.
- Data is sorted by `subject_id` and `timestamp_s` before any temporal transform.

Feature-ablation artifacts:
- `artifacts/metrics/grouped_cv_feature_ablation_summary.csv`
- `artifacts/metrics/grouped_cv_final_feature_summary.csv`
- `artifacts/figures/grouped_cv_feature_ablation_mae.png`

---

## Telemetry tables

### Interim table
File:
- `data/interim/pamap2_per_second.parquet`

Schema contract:
- `docs/schemas/interim_telemetry_schema.json`

One row per:
- subject
- second

Expected columns:
- subject_id
- timestamp
- activity_id
- heart_rate
- compact sensor summary features

### Final modeling tables (task-specific)
Files:
- `data/processed/pamap2_model_table_regression.parquet`
- `data/processed/pamap2_model_table_classification.parquet`

Schema contract:
- `docs/schemas/model_table_schema.json`

Both tables are derived from one shared upstream feature/target build.

Shared additions from Phase 3+
- baseline lag/rolling features
- targeted upgraded features (19 additional columns)
- `hr_target_30s`, `hr_target_15s`, `hr_target_next30s_mean`
- `activity_target`
- HR fill-policy metadata columns (`heart_rate_fill_strategy`, `heart_rate_observed_flag`)

Row eligibility split (2026-04-17 update):
- regression-ready table: requires selected features + selected regression target (`preferred_target_col`)
- classification-ready table: requires selected features + `activity_target` only

Why this split is methodologically correct:
- classification predicts current activity state,
- regression predicts future heart-rate targets,
- future-target missingness should not remove valid classification rows.

Core update artifacts:
- `artifacts/metrics/grouped_cv_feature_ablation_summary.csv`
- `artifacts/metrics/grouped_cv_target_comparison_summary.csv`
- `artifacts/metrics/grouped_cv_fill_sensitivity_summary.csv`
- `artifacts/metrics/grouped_cv_preferred_setup_summary.csv`

---

## Implemented target definitions

### Regression target
Default:
- heart rate shifted by -30 seconds within subject

Fallback only if necessary:
- mean heart rate across the next 30 seconds within subject

Implemented target variants for compact comparison:
- `hr_target_30s = heart_rate_bpm shifted by -30 seconds within subject`
- `hr_target_next30s_mean = mean(heart_rate_bpm at t+1 ... t+30) within subject`
- `hr_target_15s = heart_rate_bpm shifted by -15 seconds within subject`

Preferred grouped-evaluation target after ablation:
- `hr_target_next30s_mean`

Why:
- it reduced grouped held-out MAE by `2.759` versus direct `t + 30s` under the same upgraded feature set.

### Classification target
Default:
- current activity label at time t

Implemented in Phase 3:
- `activity_target = activity_id` (current activity)
- Rows used for classification require classification features and `activity_target` only.
- Classification rows are no longer filtered by regression-target availability.

### Uncertainty
- split conformal prediction intervals around the regression model output

---

## Implemented grouped evaluation strategy (2026-04-12)

The modeling workflow now uses subject-grouped cross-validation instead of one fixed validation/test split.

Implemented strategy:
- leave-one-subject-out (LOSO) grouped cross-validation
- one full held-out subject per fold
- no subject overlap between train and fold-test data
- same grouped folds for every model comparison

Current fold setup:
- subjects covered: `101` to `108`
- total folds: `8`
- train subjects per fold: `7`

Core grouped artifacts:
- `artifacts/metrics/grouped_cv_regression_fold_metrics.csv`
- `artifacts/metrics/grouped_cv_regression_summary.csv`
- `artifacts/metrics/grouped_cv_classification_fold_metrics.csv`
- `artifacts/metrics/grouped_cv_classification_summary.csv`
- `artifacts/metrics/grouped_cv_selected_model_summary.csv`

### Grouped model comparison results

Regression on preferred target `hr_target_next30s_mean` (mean MAE across LOSO folds):
- `hist_gradient_boosting`: `3.8127` (selected)
- `linear_regression`: `3.9483`
- `persistence_current_hr`: `4.1304`

Classification (mean macro F1 across LOSO folds):
- `random_forest`: `0.7437` (selected)
- `logistic_regression`: `0.7429`

Selection rule used in code and artifacts:
- regression: lowest mean MAE, tie-break by lower MAE std then lower mean RMSE
- classification: highest mean macro F1, tie-break by lower macro F1 std then higher mean accuracy

### Breakdown and uncertainty outputs

Where performance breaks is now reported with:
- by-subject tables:
	- `artifacts/metrics/grouped_cv_regression_selected_by_subject.csv`
	- `artifacts/metrics/grouped_cv_classification_selected_by_subject.csv`
- by-activity tables:
	- `artifacts/metrics/grouped_cv_regression_selected_by_activity.csv`
	- `artifacts/metrics/grouped_cv_classification_selected_by_activity.csv`
- classification per-class table:
	- `artifacts/metrics/grouped_cv_classification_selected_per_class.csv`

Grouped conformal outputs:
- `artifacts/metrics/grouped_cv_conformal_fold_summary.csv`
- `artifacts/metrics/grouped_cv_conformal_summary.csv`
- `artifacts/metrics/grouped_cv_conformal_by_subject.csv`
- `artifacts/metrics/grouped_cv_conformal_by_activity.csv`

Expanded conformal diagnostics now also include:
- all-variant comparison files:
	- `artifacts/metrics/grouped_cv_conformal_summary_all_variants.csv`
	- `artifacts/metrics/grouped_cv_conformal_variant_comparison.csv`
	- `artifacts/metrics/grouped_cv_conformal_fold_summary_all_variants.csv`
	- `artifacts/metrics/grouped_cv_conformal_by_activity_all_variants.csv`
	- `artifacts/metrics/grouped_cv_conformal_by_subject_all_variants.csv`
- residual and failure diagnostics:
	- `artifacts/metrics/grouped_cv_regression_residual_summary.csv`
	- `artifacts/metrics/grouped_cv_uncertainty_failure_by_activity.csv`
	- `artifacts/metrics/grouped_cv_uncertainty_failure_by_subject.csv`
	- `artifacts/metrics/grouped_cv_uncertainty_operating_envelope_by_activity.csv`

Conformal method policy (2026-04-13 update):
- keep global split conformal as baseline,
- evaluate activity-conditioned split conformal each run,
- select conditioned intervals only when activity-level coverage stability improves enough relative to interval-width cost,
- otherwise retain global intervals and document why.

Grouped conformal headline values:
- reported directly from `artifacts/metrics/grouped_cv_conformal_summary.csv`
- includes row-level coverage, interval width, activity-level coverage-gap stability, and variant-selection reasoning

Classification confidence and calibration outputs (2026-04-13 update):
- `artifacts/metrics/grouped_cv_classification_calibration_summary.csv`
- `artifacts/metrics/grouped_cv_classification_reliability_by_bin.csv`
- `artifacts/metrics/grouped_cv_classification_abstention_summary.csv`
- figures:
	- `artifacts/figures/grouped_cv_classification_reliability_curve.png`
	- `artifacts/figures/grouped_cv_classification_abstention_tradeoff.png`

Focused-ablation support artifacts:
- `artifacts/metrics/grouped_cv_ablation_fold_metrics.csv`
- `artifacts/metrics/grouped_cv_ablation_model_summaries.csv`
- `artifacts/figures/grouped_cv_target_comparison_mae.png`
- `artifacts/figures/grouped_cv_fill_sensitivity_mae.png`

---

## Leakage risks to watch

These are the main failure modes for this project:

1. random row-level splitting
2. rolling statistics computed across subject boundaries
3. future information leaking into feature columns
4. target leakage through incorrectly shifted heart-rate columns
5. using cleaned or filled values in a way that indirectly uses future data
6. class labels or segments bleeding across splits

Every major notebook should explicitly guard against these.

---

## Questions to answer after Phase 1

Once the audit is done, update this file with answers to:

1. Which activity classes stay in the MVP?
	- `1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17`
2. How severe is heart-rate missingness?
	- Severe at raw frequency (about 90.86% missing per row), expected from HR sampling near 9 Hz against 100 Hz IMU rows.
3. Is direct `t + 30 seconds` forecasting stable enough?
	- It is stable and usable, but grouped ablation showed lower error for the next-30s average target. The project now documents both formulations explicitly.
4. Which compact sensor features are most useful?
	- Phase 1 interim keeps `heart_rate_bpm`, 16g accelerometer magnitudes, and gyroscope magnitudes for hand/chest/ankle.
5. Does resampling to 1 second preserve enough signal for the MVP?
	- Yes for the MVP scope; 1-second aggregation produced a usable Protocol table and aligns with project constraints.