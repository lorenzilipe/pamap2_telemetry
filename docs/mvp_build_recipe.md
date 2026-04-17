# mvp build recipe

This is the canonical rerun sequence for the final project state.

## prerequisites

- Python 3.11 environment active
- PAMAP2 raw files available under `data/raw/`
- dependencies installed from `requirements.txt` (or conda `environment.yml`)

## step 1: compact ablation and table build

Run:

```bash
python scripts/compact_ablation_study.py
```

What this step does:
- rebuilds shared upstream telemetry features
- compares compact feature/target/fill variants
- writes preferred setup metadata
- writes task-specific processed tables:
  - `data/processed/pamap2_model_table_regression.parquet`
  - `data/processed/pamap2_model_table_classification.parquet`

Key output to check:
- `artifacts/metrics/grouped_cv_preferred_setup_summary.csv`

## step 2: grouped evaluation and diagnostics

Run:

```bash
python scripts/grouped_evaluation.py
```

What this step does:
- reads preferred regression target from step 1 metadata
- runs leave-one-subject-out grouped CV for both tasks
- writes selected-model summaries and fold metrics
- writes conformal, calibration, and abstention diagnostics

Key outputs to check:
- `artifacts/metrics/grouped_cv_selected_model_summary.csv`
- `artifacts/metrics/grouped_cv_regression_summary.csv`
- `artifacts/metrics/grouped_cv_classification_summary.csv`
- `artifacts/metrics/grouped_cv_conformal_summary.csv`
- `artifacts/metrics/grouped_cv_classification_calibration_summary.csv`

## step 3: selected model records

Run:

```bash
python scripts/write_experiment_records.py
```

What this step does:
- refreshes task-specific selected model records
- writes canonical tracked copies in `docs/model_records/`

Outputs:
- `docs/model_records/regression_selected_model_record.json`
- `docs/model_records/classification_selected_model_record.json`

## expected workflow properties

- no subject leakage: all temporal transforms are subject-local
- no silent target drift: grouped evaluation reads preferred target metadata
- no cross-task row mismatch: regression and classification use separate task tables

## quick verification checklist

After a full rerun, confirm:
- preferred setup file exists and has `preferred_target_col`
- selected model summary includes one regression and one classification selection
- conformal summary has a preferred interval variant
- model records in `docs/model_records/` match grouped summary metrics

## notebooks and scripts

The notebooks and scripts should tell the same story:
- notebook 01: ingest and audit
- notebook 02: feature pipeline and ablation
- notebook 03: grouped modeling and uncertainty
- scripts: thin wrappers around `src/pamap2_telemetry/`

If notebook outputs and script outputs disagree, treat script outputs as the canonical rerun reference and reconcile notebooks.
