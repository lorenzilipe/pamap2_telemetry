# decision log

This file records the decisions that define the current project state.

## 2026-04-07 dataset and project shape

### decision
Use PAMAP2 as the core dataset and keep the project notebook-first and local-only.

### why
It supports a clear wearable telemetry story without infrastructure overhead.

### consequences
The repository focuses on data pipeline quality, grouped evaluation, and clear evidence artifacts.

## 2026-04-07 task set

### decision
Use one shared telemetry pipeline to support:
1. heart-rate forecasting,
2. current activity classification,
3. regression uncertainty intervals.

### why
This keeps the project coherent instead of splitting it into unrelated analyses.

### consequences
Feature engineering and evaluation must serve both regression and classification.

## 2026-04-07 protocol-first ingest and activity subset

### decision
Use Protocol sessions as the canonical source and keep activity IDs:
`1,2,3,4,5,6,7,12,13,16,17`.

### why
Protocol data is the most consistent source across subjects, and the kept activities have stronger support.

### consequences
Rows with `activity_id = 0` are excluded from supervised tasks.

## 2026-04-12 grouped evaluation baseline

### decision
Use leave-one-subject-out grouped cross-validation as the default evaluation strategy.

### why
Single held-out-subject selection was too fragile and less defensible.

### alternatives rejected
- random row-level splits
- one fixed validation subject for model choice

### consequences
Model ranking and uncertainty diagnostics are based on grouped folds.

## 2026-04-12 compact ablation and preferred setup

### decision
Run compact ablations for feature set, target definition, and HR fill strategy, then select one preferred setup.

### why
These choices materially affect realism and error levels, and needed explicit evidence.

### consequences
Current preferred setup:
- feature set: `upgraded`
- fill strategy: `current_ffill`
- regression target: `hr_target_next30s_mean`

## 2026-04-13 package boundary and contracts

### decision
Move reusable logic into `src/pamap2_telemetry/` and keep notebooks/scripts as orchestration.

### why
This improves rerun consistency while preserving notebook readability.

### consequences
Data contracts and selected-model records were added as explicit project artifacts.

## 2026-04-14 preferred-target rerun consistency

### decision
Make grouped evaluation read `preferred_target_col` from preferred setup artifacts, and fail fast when missing or inconsistent.

### why
Hardcoded fallback targets can silently drift from ablation conclusions.

### consequences
Standalone reruns and notebook reruns stay aligned to the same regression target.

## 2026-04-17 task-specific downstream tables

### decision
Keep one shared upstream feature build, but split downstream processed tables into regression-ready and classification-ready tables.

### why
Classification should not lose valid rows because regression targets are unavailable.

### consequences
Grouped evaluation now reads:
- `data/processed/pamap2_model_table_regression.parquet`
- `data/processed/pamap2_model_table_classification.parquet`

## 2026-04-17 online-ish delayed-HR sensitivity check

### decision
Add one lightweight stress mode, `onlineish_hr_delay_5s`, that delays HR availability before HR-derived feature generation.

### why
Offline assumptions about immediate HR access can inflate regression performance.

### consequences
The repo reports how metrics shift under a simple delayed-input scenario without expanding into streaming infrastructure.
