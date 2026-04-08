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