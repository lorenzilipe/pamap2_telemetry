# mvp scope

## in scope

- one shared telemetry pipeline from PAMAP2 Protocol data
- one regression task: near-future heart-rate forecasting
- one classification task: current activity-state prediction
- one uncertainty layer: split conformal intervals for regression
- grouped leave-one-subject-out evaluation
- compact model set and compact feature set
- saved metric artifacts and selected model records

## canonical targets

- regression preferred target: `hr_target_next30s_mean`
- classification target: `activity_target`

Alternative regression targets can be compared in ablation, but grouped evaluation should run on the preferred target selected by artifact metadata.

## data representation

- 1-second telemetry rows per subject
- subject-local lag and rolling features only
- separate processed tables for regression and classification row eligibility

## required deliverables

- notebooks:
  - `notebooks/01_ingest_and_audit.ipynb`
  - `notebooks/02_feature_pipeline.ipynb`
  - `notebooks/03_modeling_and_uncertainty.ipynb`
- reusable package code in `src/pamap2_telemetry/`
- metrics and figures under `artifacts/`
- contracts and decision docs under `docs/`

## non-goals

- API, dashboard, cloud deployment, or streaming runtime
- deep learning or large hyperparameter sweeps
- experiment tracking platform integration
- broad infrastructure work outside the modeling pipeline

## scope guardrail

When choosing between alternatives, prefer the smallest change that improves correctness, interpretability, or evaluation realism.
