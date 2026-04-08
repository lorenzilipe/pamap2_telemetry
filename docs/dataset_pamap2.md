# PAMAP2 dataset working notes

## Purpose of this document

This file is the project-specific source of truth for how PAMAP2 is being interpreted and used in this repository.

It should answer:
- what raw data fields matter,
- how the dataset is being cleaned,
- how timestamps and labels are handled,
- how the 1-second telemetry table is built,
- how final modeling features and targets are defined.

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

## Planned compact telemetry feature set

The MVP should use a compact feature set rather than every raw axis directly.

### Core numeric signals
- heart rate
- accelerometer magnitude by body location
- gyroscope magnitude by body location
- optional temperature by body location

### Planned derived features
For each selected numeric signal:
- current value
- lag 1 second
- lag 5 seconds
- rolling mean over last 5 seconds
- rolling std over last 5 seconds
- rolling mean over last 10 seconds
- rolling std over last 10 seconds
- short-term change from rolling mean

### Planned ablation reminder (run later)
To confirm whether axis information is worth the extra complexity, run this deferred ablation in Phase 3/4.

Feature setups:
- Setup A: magnitude-only features (current MVP baseline)
- Setup B: magnitude features plus a small axis subset

Hold constant during ablation:
- subject-held-out split
- target definitions
- model families and random seed

Compare:
- regression: MAE, RMSE
- classification: macro F1, confusion matrix (check stair-direction confusion)
- uncertainty: empirical coverage, average interval width

Adoption rule:
- keep Setup B only if it provides clear held-out gains without making the notebook much harder to explain
- otherwise keep Setup A for MVP defensibility

---

## Planned telemetry table

### Interim table
File:
- `data/interim/pamap2_per_second.parquet`

One row per:
- subject
- second

Expected columns:
- subject_id
- timestamp
- activity_id
- heart_rate
- compact sensor summary features

### Final modeling table
File:
- `data/processed/pamap2_model_table.parquet`

Expected additions:
- lagged features
- rolling features
- `hr_target_30s`
- `activity_target`

---

## Planned target definitions

### Regression target
Default:
- heart rate shifted by -30 seconds within subject

Fallback only if necessary:
- mean heart rate across the next 30 seconds within subject

### Classification target
Default:
- current activity label at time t

### Uncertainty
- split conformal prediction intervals around the regression model output

---

## Planned split strategy

Use subject-held-out splits.

Record final chosen split here after implementation:

- train subjects: 101, 102, 103, 104, 105, 106
- validation subjects: 107
- test subjects: 108
- excluded from default split due low coverage: 109

Reason for split choice:
- Preserves subject-held-out design while avoiding unstable evaluation from the very low-coverage subject 109.
- Keeps one validation subject and one test subject from unseen people.

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
	- No blocker found in Phase 1. Proceed with direct shift in Phase 2/3 and validate on held-out subjects.
4. Which compact sensor features are most useful?
	- Phase 1 interim keeps `heart_rate_bpm`, 16g accelerometer magnitudes, and gyroscope magnitudes for hand/chest/ankle.
5. Does resampling to 1 second preserve enough signal for the MVP?
	- Yes for the MVP scope; 1-second aggregation produced a usable Protocol table and aligns with project constraints.