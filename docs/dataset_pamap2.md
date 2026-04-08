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
- raw files location:
- subject file naming pattern:
- documentation file path:

### Core raw fields to identify
- timestamp column:
- subject ID source:
- activity label column:
- heart-rate column:
- hand accelerometer columns:
- chest accelerometer columns:
- ankle accelerometer columns:
- hand gyroscope columns:
- chest gyroscope columns:
- ankle gyroscope columns:
- temperature columns, if used:

### Initial audit outputs to record
- rows per subject:
- heart-rate missingness by subject:
- activity counts overall:
- activity counts by subject:
- invalid or unlabeled row counts:
- obvious anomalies or caveats:

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
- to be filled after audit

Reason:
- to be filled after audit

### Heart-rate handling
Current default plan:
- inspect missingness carefully during audit
- after resampling to 1-second rows, forward-fill within subject
- optionally backfill only short gaps if needed
- do not use complex imputation in v1

If a different rule is adopted, document it here.

### Time handling
- sort within subject by timestamp before any temporal operation
- resample to 1-second intervals for the main telemetry table
- use only past information to create features
- use future information only to define targets

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

- train subjects:
- validation subjects:
- test subjects:

Reason for split choice:
- to be filled after audit and initial modeling

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
2. How severe is heart-rate missingness?
3. Is direct `t + 30 seconds` forecasting stable enough?
4. Which compact sensor features are most useful?
5. Does resampling to 1 second preserve enough signal for the MVP?