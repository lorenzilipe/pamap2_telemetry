# Decision log

This file records project decisions so that future work sessions stay consistent.

Use this format for new entries:

## [date] Decision title
### Decision
### Why
### Alternatives rejected
### Consequences

---

## 2026-04-07 Dataset choice
### Decision
Use PAMAP2 as the primary dataset for the wearable telemetry MVP.

### Why
It best supports a Garmin-adjacent story around wearable sensor analytics, time-series forecasting, and activity-state classification while staying manageable for a lean notebook-first project.

### Alternatives rejected
- WESAD: strong physiology and stress angle, but less directly aligned with the desired wearable fitness telemetry framing for this MVP
- C-MAPSS: good telemetry dataset, but industrial rather than wearable

### Consequences
The project should foreground wearable telemetry, sensor features, heart-rate forecasting, and activity classification.

---

## 2026-04-07 MVP scope choice
### Decision
Build an ultra-lean, notebook-only MVP.

### Why
The immediate goal is to create a strong, defensible portfolio project quickly without getting trapped in infrastructure work.

### Alternatives rejected
- API-first build
- dashboard-first build
- production-style deployment
- cloud architecture

### Consequences
The project should focus on data ingestion, feature engineering, modeling, uncertainty estimation, and clear evaluation.

---

## 2026-04-07 Primary modeling tasks
### Decision
Use one shared dataset pipeline to support:
1. heart-rate forecasting 30 seconds ahead,
2. current activity-state classification,
3. lightweight regression prediction intervals.

### Why
This creates one coherent telemetry project instead of multiple disconnected analyses.

### Alternatives rejected
- regression only
- classification only
- anomaly detection as the main task
- future activity-transition prediction as the main classification target

### Consequences
The feature pipeline must support both regression and classification while remaining compact and interpretable.

---

## 2026-04-07 Evaluation choice
### Decision
Use subject-held-out splits as the default evaluation design.

### Why
This is more realistic and more interview-defensible than random row-level splitting.

### Alternatives rejected
- random train/test row split
- purely time-based split across pooled subjects without subject separation

### Consequences
All feature engineering and target generation must be done in a way that respects subject boundaries.

---

## 2026-04-07 Modeling complexity guardrail
### Decision
Do not use deep learning or broad Dask integration in v1 unless there is a clear and specific reason.

### Why
The highest-ROI MVP is a clean, readable, well-evaluated sklearn-based pipeline.

### Alternatives rejected
- deep neural sequence models
- full distributed-computing redesign
- large hyperparameter sweeps

### Consequences
The project should prioritize speed, clarity, and correctness over technical maximalism.

---

## 2026-04-07 Collaboration style
### Decision
Work in small, scoped tasks with explicit planning before implementation.

### Why
This reduces scope drift and makes it easier to keep the project clean and explainable.

### Alternatives rejected
- giant open-ended coding sessions
- mixing planning, implementation, and interpretation all at once

### Consequences
Each work session should focus on one phase or one subtask at a time.

---

## 2026-04-07 Raw package restructuring approach
### Decision
Keep the original PAMAP2 raw files immutable and add a tidy metadata/documentation layer inside `data/raw/pamap2+physical+activity+monitoring/`.

### Why
This preserves provenance while making ingestion and auditing easier to reproduce. The metadata layer captures schema, labels, subject metadata, file manifests, and a concise documentation digest from raw PDFs.

### Alternatives rejected
- Physically renaming or moving raw source files in v1
- Leaving raw docs only as PDFs without machine-readable extraction

### Consequences
Phase 1 now has a neater structure without breaking source integrity. Notebook code continues reading the original dataset paths while docs and manifests provide cleaner references.

---

## 2026-04-07 Phase 1 ingest scope and activity subset
### Decision
Use Protocol sessions as the primary Phase 1 ingest/audit scope, keep Optional sessions as supplemental inventory only, and retain activity IDs `1,2,3,4,5,6,7,12,13,16,17` for the MVP.

### Why
Protocol is the canonical shared collection across subjects and avoids mixing supplemental sessions that overlap the same people. The kept activity set was chosen with transparent thresholds (at least 6 subjects and at least 90,000 rows) to keep high-support classes.

### Alternatives rejected
- Merging Protocol and Optional sessions in primary audit metrics
- Keeping all labeled classes including low-support classes (for example rope jumping)

### Consequences
The interim Phase 1 table is cleaner and more stable for v1 modeling. Leakage risk from Protocol/Optional overlap is reduced, and class support is better balanced for classification.

---

## 2026-04-07 Subject 109 split handling
### Decision
Exclude subject 109 from the default train/validation/test split due very low protocol coverage.

### Why
Subject 109 contains far fewer rows than other protocol subjects, which can destabilize evaluation and split balance if treated as a standard held-out subject.

### Alternatives rejected
- Keeping subject 109 as a standard test subject
- Dropping subject 109 from raw audit entirely

### Consequences
Default split guidance is now train `101-106`, validation `107`, test `108`, with subject `109` preserved for optional sensitivity checks.

---

## 2026-04-08 Phase 1 audit closeout and strict Phase 2 verification
### Decision
Close remaining Phase 1 audit gaps in notebook 01 and mark strict recipe Phase 2 as complete only after explicit validation checks pass on the regenerated interim table.

### Why
The project already had working Phase 2 logic in notebook 01, but audit completeness and completion evidence were uneven. Adding missingness and summary artifacts plus time-series audit plots improves traceability and interview defensibility. Explicit Phase 2 checks reduce ambiguity before moving to Phase 3.

### Alternatives rejected
- Treating Phase 1 as complete without adding missing audit artifacts
- Advancing directly into Phase 3 while Phase 2 completion evidence remained implicit
- Refactoring Phase 2 code into notebook 02 before first finishing strict completion checks

### Consequences
- New audit artifacts were added:
	- `artifacts/metrics/phase1_protocol_column_missingness.csv`
	- `artifacts/metrics/phase1_protocol_hr_summary_stats.csv`
	- `artifacts/figures/phase1_protocol_hr_timeseries_subject104.png`
	- `artifacts/figures/phase1_protocol_hand_acc_timeseries_subject104.png`
- Strict Phase 2 validation artifacts were added:
	- `artifacts/metrics/phase2_strict_validation_checks.csv`
	- `artifacts/metrics/phase2_interim_hr_missingness_by_subject.csv`
- Phase 2 is now explicitly verified as complete with schema, uniqueness, sort-order, activity-set, and post-fill heart-rate checks.

---

## 2026-04-08 Phase 3 feature pipeline completion
### Decision
Complete Phase 3 as a strict feature-engineering step using the compact magnitude-based signal set only, and generate the final supervised table at `data/processed/pamap2_model_table.parquet` plus Phase 3 validation artifacts.

### Why
This matches the MVP build recipe and keeps scope focused: create past-only lagged and rolling features, define `hr_target_30s` and `activity_target`, and avoid drifting early into modeling or expanded feature families. The compact set is easier to explain and already aligned with the project's interview-defensible story.

### Alternatives rejected
- Adding temperature features in Phase 3 by default
- Expanding to axis-level feature sets before baseline Phase 4/5 evaluation
- Starting Phase 4 EDA in the same implementation pass
- Switching the regression target to a next-30s average before testing the direct shift target

### Consequences
- Phase 3 notebook implementation added in `notebooks/02_feature_pipeline.ipynb`.
- Final modeling table generated with 18,627 rows and 63 columns.
- Per-subject row loss from windowing plus future shift is 39 rows each (8 subjects, 312 total).
- Phase 3 artifacts added:
	- `artifacts/metrics/phase3_row_retention_by_subject.csv`
	- `artifacts/metrics/phase3_feature_missingness_post_clean.csv`
	- `artifacts/metrics/phase3_target_summary.csv`
- `docs/dataset_pamap2.md` now records implemented (not just planned) Phase 3 feature and target definitions.

---

## 2026-04-08 Cross-platform environment workflow
### Decision
Use machine-local Python environments and a shared kernel name (`Python (pamap2-telemetry)`) across Windows and macOS, with setup scripts per operating system.

### Why
Virtual environments are not portable across operating systems. Syncing a macOS `.venv` into Windows caused kernel startup failures. A local-per-machine environment keeps notebook execution stable while preserving one shared codebase.

### Alternatives rejected
- Sharing one `.venv` folder across macOS and Windows through cloud sync
- Keeping OS-specific interpreter paths in repository settings

### Consequences
- Added setup scripts:
	- `scripts/setup_windows.ps1`
	- `scripts/setup_mac.sh`
- Added transition instructions in `README.md`.
- Added `.venv_mac/` and `.venv_win/` to `.gitignore`.

---

## 2026-04-08 Phase 4 to Phase 8 implementation complete
### Decision
Complete Phases 4-8 in notebook 03 with subject-held-out evaluation, baseline and stronger models, and split conformal intervals.

### Why
This closes the core MVP modeling scope while staying lean, transparent, and interview-defensible. The workflow now includes EDA evidence, fixed splits, baseline comparisons, stronger model checks, and calibrated uncertainty.

### Alternatives rejected
- Skipping baseline models and moving directly to stronger models
- Changing the split strategy midstream
- Using complex uncertainty methods beyond split conformal for v1

### Consequences
- Notebook implementation completed in `notebooks/03_modeling_and_uncertainty.ipynb`.
- New metrics artifacts added:
	- `artifacts/metrics/phase4_target_difficulty_checks.csv`
	- `artifacts/metrics/phase5_split_summary.csv`
	- `artifacts/metrics/phase6_baseline_metrics.csv`
	- `artifacts/metrics/phase7_regression_tuning_results.csv`
	- `artifacts/metrics/phase7_classification_tuning_results.csv`
	- `artifacts/metrics/phase7_stronger_model_metrics.csv`
	- `artifacts/metrics/phase6_7_all_model_metrics.csv`
	- `artifacts/metrics/phase7_final_model_selection.csv`
	- `artifacts/metrics/phase8_conformal_summary.csv`
	- `artifacts/metrics/phase8_conformal_by_activity.csv`
	- `artifacts/metrics/phase8_conformal_test_predictions.csv`
- New figure artifacts added:
	- `artifacts/figures/phase4_regression_heart_rate_distribution.png`
	- `artifacts/figures/phase4_regression_hr_by_activity_boxplot.png`
	- `artifacts/figures/phase4_regression_subject_timeseries.png`
	- `artifacts/figures/phase4_regression_top_feature_corr_heatmap.png`
	- `artifacts/figures/phase4_class_balance.png`
	- `artifacts/figures/phase4_sensor_trace_by_activity.png`
	- `artifacts/figures/phase4_motion_distribution_by_activity.png`
	- `artifacts/figures/phase7_confusion_matrix_test.png`
	- `artifacts/figures/phase8_interval_example_test.png`
	- `artifacts/figures/phase8_coverage_by_activity.png`
- Saved final trainable models in `artifacts/models/`:
	- `phase7_final_regression_model.joblib`
	- `phase7_final_classification_model.joblib`