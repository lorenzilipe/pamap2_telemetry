# PAMAP2 Wearable Telemetry MVP

This repository contains an ultra-lean, notebook-first data science project built on the PAMAP2 wearable telemetry dataset. The goal is to create a strong, interview-defensible portfolio project that demonstrates time-series forecasting, telemetry and sensor analytics, data pipeline thinking, clean evaluation, and lightweight uncertainty estimation.

The MVP is intentionally scoped to stay small enough to complete quickly while still showing end-to-end rigor. The project uses one shared dataset pipeline to support three related tasks:
1. forecast heart rate 30 seconds ahead,
2. classify current activity state,
3. produce lightweight prediction intervals for the regression task.

This project is being built for portfolio and resume value, especially for data science roles that care about telemetry, forecasting, model evaluation, and practical ML workflows. It is notebook-only in v1. There is no cloud, no API, no dashboard, and no deep learning unless explicitly added later.

## Core docs

Read these in order:

1. `docs/project_overview.md`
2. `docs/mvp_scope.md`
3. `docs/dataset_pamap2.md`
4. `docs/mvp_build_recipe.md`
5. `docs/copilot_workflow.md`

## Repository layout

- `docs/` — source-of-truth documentation for scope, workflow, and decisions
- `notebooks/` — main build notebooks
- `data/raw/` — original dataset files
- `data/interim/` — intermediate tables, especially the 1-second telemetry table
- `data/processed/` — final supervised-learning tables
- `artifacts/figures/` — saved plots
- `artifacts/metrics/` — saved metrics tables
- `artifacts/models/` — optional saved models

## Cross-platform environment setup

This project is edited on both Windows and macOS. To avoid kernel failures, keep Python environments machine-local and never sync one OS environment to the other.

Use the setup scripts from repository root:

- Windows (PowerShell):
  - `./scripts/setup_windows.ps1`
- macOS (bash):
  - `bash ./scripts/setup_mac.sh`

What these scripts do:
1. detect and rename incompatible synced `.venv` folders when needed,
2. create a local machine environment outside the synced repo,
3. install dependencies from `requirements.txt`,
4. register a shared notebook kernel name: `Python (pamap2-telemetry)`.

### Transition checklist between computers

1. Save and commit notebook/code changes.
2. Push to Git and wait for cloud sync to finish.
3. On the next machine, pull latest changes.
4. Run the local setup script for that OS if needed.
5. In VS Code notebooks, select kernel `Python (pamap2-telemetry)`.

## Initial notebook plan

- `01_ingest_and_audit.ipynb`
  - load raw PAMAP2 files
  - inspect columns, missingness, activity labels, and subject coverage
  - decide which activities to keep for the MVP

- `02_feature_pipeline.ipynb`
  - compute sensor magnitudes
  - resample to a 1-second telemetry table
  - create lagged and rolling features
  - define regression and classification targets

- `03_modeling_and_uncertainty.ipynb`
  - run leave-one-subject-out grouped model comparison
  - compare a compact model set per task
  - select final models from grouped CV summaries
  - add grouped conformal prediction intervals
  - generate fold, subject, and activity breakdown artifacts

## Evaluation story

The final modeling workflow is subject-aware end to end:

- model selection uses leave-one-subject-out cross-validation
- fold metrics are saved for every candidate model
- aggregate metrics include mean, standard deviation, min, and max across folds
- final model decisions are based on grouped CV summaries, not one validation subject
- error analysis is reported by subject, by activity, and by class (for classification)

Grouped evaluation artifacts use `grouped_cv_` prefixes under `artifacts/metrics/` and `artifacts/figures/`.

## Uncertainty and confidence outputs

The uncertainty layer is now diagnostic, not decorative.

Regression uncertainty outputs include:

- overall conformal coverage and interval width,
- by-activity and by-subject coverage and interval width,
- global vs activity-conditioned conformal comparison,
- large-error and interval-failure hotspot tables,
- residual distribution and residual summary statistics,
- an operating-envelope table that flags fragile activity regimes.

Key metric tables:

- `artifacts/metrics/grouped_cv_conformal_summary.csv`
- `artifacts/metrics/grouped_cv_conformal_by_activity.csv`
- `artifacts/metrics/grouped_cv_conformal_by_subject.csv`
- `artifacts/metrics/grouped_cv_uncertainty_failure_by_activity.csv`
- `artifacts/metrics/grouped_cv_uncertainty_failure_by_subject.csv`
- `artifacts/metrics/grouped_cv_regression_residual_summary.csv`
- `artifacts/metrics/grouped_cv_uncertainty_operating_envelope_by_activity.csv`

Key uncertainty figures:

- `artifacts/figures/grouped_cv_conformal_coverage_by_activity.png`
- `artifacts/figures/grouped_cv_conformal_interval_failure_by_activity.png`
- `artifacts/figures/grouped_cv_regression_residual_distribution.png`

Classification is also evaluated as a confidence-aware decision component.

Key confidence/calibration outputs:

- `artifacts/metrics/grouped_cv_classification_calibration_summary.csv`
- `artifacts/metrics/grouped_cv_classification_reliability_by_bin.csv`
- `artifacts/metrics/grouped_cv_classification_abstention_summary.csv`
- `artifacts/figures/grouped_cv_classification_reliability_curve.png`
- `artifacts/figures/grouped_cv_classification_abstention_tradeoff.png`

## Current limitations and operating envelope

What current activity classification demonstrates:

- strong subject-held-out macro F1 for a compact model set,
- useful probability signals for confidence-based filtering.

What it does not demonstrate:

- robust calibration under domain shift (new users/devices),
- production-grade behavior under sensor drift and missing channels.

Heart-rate forecasting interpretation:

- forecasting remains partially persistence-driven,
- gains beyond persistence are present but not uniform across activities.

Where uncertainty is weakest:

- activities and subjects with higher interval-failure rates and heavier residual tails,
- activity regimes where activity-conditioned calibration still falls back to global margins due low calibration support.

Current run highlights (from grouped metrics):

- compact changes reduced failure pressure before intervaling:
  - upgraded features improved grouped mean MAE by about `0.364` bpm versus baseline,
  - switching to `hr_target_next30s_mean` improved grouped mean MAE by about `2.720` bpm versus direct `t+30s`,
  - after those changes, interval failures are concentrated mostly in stair activities rather than broadly distributed.
- weakest interval coverage activity: `ascending_stairs` (coverage about `0.573`),
- next weakest interval coverage activity: `descending_stairs` (coverage about `0.798`),
- global conformal remained preferred over activity-conditioned because activity-level coverage-gap gain was small (`+0.009` weighted abs-gap improvement) and did not justify switching policy.

Current confidence diagnostics highlights:

- classification calibration is imperfect but usable (`ECE_10` about `0.073`, multiclass Brier about `0.306`),
- average confidence is higher than realized accuracy by about `0.073`, so overconfidence should be communicated,
- abstention tradeoff is practical: threshold `0.8` keeps about `70.8%` of rows and raises retained accuracy to about `0.898`.

Next production-minded improvements (kept lightweight):

- gather more calibration support for hard activities,
- add a confidence gate policy for classification decisions,
- target feature updates toward recurring failure hotspots rather than broad model complexity.

## Principles

- Keep scope disciplined.
- Prefer simple, correct baselines over complexity.
- Avoid leakage.
- Document every non-obvious decision.
- Optimize for both resume value and interview defensibility.
