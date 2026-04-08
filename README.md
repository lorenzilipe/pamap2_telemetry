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
  - build subject-held-out splits
  - run baselines and stronger sklearn models
  - add split conformal prediction intervals
  - generate final plots and metrics

## Principles

- Keep scope disciplined.
- Prefer simple, correct baselines over complexity.
- Avoid leakage.
- Document every non-obvious decision.
- Optimize for both resume value and interview defensibility.