# project overview

## purpose

This project is a compact wearable telemetry pipeline built on PAMAP2.

It was designed to show end-to-end applied ML work that is:
- technically honest
- easy to explain
- realistic about generalization limits

## core question

Given recent wearable history, can we:
1. forecast near-future heart rate,
2. classify current activity,
3. quantify uncertainty in the forecast?

## what was built

- Protocol-only PAMAP2 ingest and audit
- 1-second subject-level telemetry table
- subject-local lagged and rolling features
- grouped leave-one-subject-out evaluation
- split conformal regression intervals
- confidence diagnostics for classification

## modeling choices

The model set is intentionally small:
- regression: persistence, linear regression, hist gradient boosting
- classification: logistic regression, random forest

Reason: this repo prioritizes clean methodology and evidence over model breadth.

## evaluation standard

All model comparisons use subject-grouped cross-validation.

This avoids optimistic random-row splits and keeps the main claim tied to cross-subject behavior.

## final claim this project supports

A lean telemetry pipeline can produce useful short-horizon forecasting and activity classification results under grouped evaluation, with uncertainty diagnostics that reveal where performance is stable and where it is fragile.

## boundaries

Out of scope in this repo:
- deployment infrastructure
- streaming system implementation
- online learning
- large model zoo or deep learning

These are deliberate scope choices, not missing implementation work.
