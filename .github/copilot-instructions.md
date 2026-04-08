# Copilot instructions for this repository

## Read this first

This repository is a notebook-first wearable telemetry data science project using the PAMAP2 dataset. The project is meant to demonstrate:

- time-series forecasting
- telemetry and sensor analytics
- data pipelines
- clean evaluation
- lightweight uncertainty estimation

This is a deliberately lean MVP. Do not turn it into a large platform or a research rabbit hole unless explicitly asked.

Before proposing or implementing any non-trivial change, read these files:

1. `docs/project_overview.md`
2. `docs/mvp_scope.md`
3. `docs/dataset_pamap2.md`
4. `docs/mvp_build_recipe.md`
5. `docs/decision_log.md`

If code or comments conflict with these docs, treat the docs as the source of truth unless the user explicitly changes scope.

---

## Current MVP tasks

This MVP currently supports three tasks from one shared dataset pipeline:

1. Forecast heart rate 30 seconds ahead.
2. Classify current activity state.
3. Produce lightweight prediction intervals for the heart-rate forecast.

---

## Current scope constraints

Keep the project within these boundaries unless explicitly told otherwise:

- notebook-only
- local-only
- Python only
- no cloud
- no paid services
- no FastAPI in v1
- no dashboard in v1
- no deep learning in v1
- no MLOps platform tooling in v1
- no Dask unless it adds real value
- no exhaustive hyperparameter tuning
- no complicated abstraction layers

Preferred stack:

- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- pyarrow
- jupyter

Optional only if justified:

- Dask

---

## Coding preferences

Follow these preferences consistently:

- Prefer simple, readable, explicit code.
- Keep functions small and focused.
- Add comments generously, especially where reasoning matters.
- Write code and comments so that someone without wearable or telemetry domain knowledge can still follow what is happening and why.
- Avoid hidden magic and unnecessary abstraction.
- Prefer transparent transformations over clever one-liners.
- For non-trivial logic, explain the assumptions near the code.

When writing analysis code:

- guard against leakage
- sort by subject and timestamp before rolling or shifting
- keep all rolling, lagging, and target generation subject-local
- use subject-held-out evaluation unless the user explicitly changes that
- document every cleaning rule and modeling decision

---

## Collaboration style

Use this behavior in chat and agent workflows:

- plan before implementing
- for non-trivial changes, first summarize the exact files to edit and the exact steps
- break work into the smallest sensible next task
- when assumptions are missing, ask concise clarification questions
- when scope starts to drift, say so clearly
- optimize for resume value and interview defensibility, not novelty for its own sake

For code reviews and suggestions:

- point out leakage risks
- point out scope creep
- point out weak evaluation design
- point out overstated claims
- prefer practical suggestions over abstract advice

---

## Project-specific modeling guidance

Use these defaults unless explicitly changed:

### Regression task
- target: heart rate 30 seconds ahead

### Classification task
- target: current activity state

### Uncertainty task
- method: split conformal prediction on top of the final regression model

### Data representation
- use a 1-second telemetry table
- create lagged and rolling features from recent history
- use a compact, interpretable feature set
- prefer sensor magnitudes and summary features over huge raw-axis expansions in the MVP

### Evaluation
- subject-held-out splits
- regression metrics: MAE, RMSE, optional R^2
- interval metrics: empirical coverage and average interval width
- classification metrics: accuracy, macro F1, confusion matrix

---

## Documentation rules

Whenever you help implement a meaningful change, also help keep docs current.

At minimum, update whichever of these is affected:

- `docs/dataset_pamap2.md`
- `docs/decision_log.md`
- `docs/mvp_scope.md`

If a decision changes scope, record:
- what changed
- why
- what was rejected instead

---

## What success looks like

The final MVP should support this honest project summary:

"I built a notebook-based wearable telemetry pipeline on PAMAP2 that cleaned raw sensor streams, built time-windowed features, forecasted heart rate 30 seconds ahead, classified activity state, and generated calibrated prediction intervals using split conformal evaluation under a subject-held-out split."

Help move the codebase toward that outcome.