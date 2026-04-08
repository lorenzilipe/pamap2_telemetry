# Project overview

## Project name

PAMAP2 Wearable Telemetry MVP

## Purpose

This project is a lean, notebook-first portfolio project built to demonstrate strong applied data science skills on wearable sensor data. The project should signal that I can work with multivariate telemetry, build a clean data pipeline, formulate sensible prediction tasks, evaluate models carefully, and communicate practical conclusions.

The portfolio goal is not just to show that I can train a model. The goal is to show that I can take raw wearable data and turn it into a compact, defensible end-to-end workflow that looks like real data science work.

## Why this project exists

I want a project that strongly supports applications for data science roles that value:

- time-series forecasting
- telemetry or sensor analytics
- practical machine learning
- clean data preprocessing
- robust evaluation
- light uncertainty estimation
- clear communication of results

This project is meant to add a stronger wearable and telemetry signal than a generic notebook analysis would.

## Why PAMAP2

PAMAP2 is a good fit because it supports a clean wearable telemetry story. It contains multivariate sensor streams and activity labels, which lets one shared dataset pipeline support both regression and classification. It also naturally supports short-horizon forecasting tasks where recent sensor history is used to predict a near-future physiological signal.

## What this MVP is trying to prove

This MVP should show that I can:

1. ingest and audit raw wearable sensor data,
2. turn raw signals into a usable telemetry table,
3. engineer time-aware features without leakage,
4. build forecasting and classification models,
5. add a simple uncertainty layer,
6. evaluate results under a subject-held-out split,
7. present conclusions clearly and honestly.

## High-level project story

The project uses PAMAP2 to build a shared 1-second telemetry table across subjects. From that table, it creates lagged and rolling features from recent sensor history. Those features are then used for three related tasks:

- forecast heart rate 30 seconds ahead,
- classify current activity state,
- generate prediction intervals for the heart-rate forecast.

This keeps the project coherent. Instead of looking like three unrelated notebook exercises, it looks like one compact telemetry pipeline that supports multiple model outputs.

## Architecture at a glance

### Stage 1: ingest and audit
- load raw subject files
- inspect columns, missingness, activity labels, and subject coverage
- choose the subset of activities to keep in the MVP

### Stage 2: build telemetry table
- compute compact sensor summary features
- resample to a 1-second table
- clean missing values using simple documented rules

### Stage 3: feature engineering
- create lags and rolling-window summaries
- define future heart-rate target
- define current activity target

### Stage 4: modeling and evaluation
- split by subject
- run simple baselines
- run one stronger sklearn regressor
- run one stronger sklearn classifier
- add split conformal intervals for regression

### Stage 5: reporting
- save metrics
- save plots
- write concise conclusions and limitations

## What is out of scope for v1

The following are explicitly out of scope unless I later decide to add them:

- web app
- API
- cloud deployment
- streaming infrastructure
- deep learning
- large-scale distributed processing
- experiment tracking platforms
- automated retraining
- online inference
- production monitoring systems

The MVP should be small, clean, and interview-defensible.

## Success criteria

This project is successful if it produces:

- one clean telemetry modeling table
- one subject-held-out evaluation workflow
- one honest regression result
- one honest classification result
- one uncertainty result with empirical coverage
- one notebook story that is easy to explain in an interview

## Resume framing target

The eventual resume framing should emphasize:

- wearable telemetry pipeline
- multivariate time-series feature engineering
- short-horizon forecasting
- activity-state classification
- subject-held-out evaluation
- lightweight uncertainty estimation

The resume framing should not overstate this as a production system, deployed platform, or finished product.