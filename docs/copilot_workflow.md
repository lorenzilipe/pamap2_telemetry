# Copilot workflow for this project

## Purpose of this document

This file explains how to work inside VS Code with GitHub Copilot on this project without losing project context or drifting off scope.

The main principle is simple:

Put stable project memory in repository docs, then make Copilot read those docs on each meaningful task.

The docs in this repository are the project's persistent memory. The most important ones are:

1. `docs/project_overview.md`
2. `docs/mvp_scope.md`
3. `docs/dataset_pamap2.md`
4. `docs/mvp_build_recipe.md`
5. `docs/decision_log.md`

---

## Default workflow

Use three different styles of Copilot interaction.

### 1. Planning
Use this when:
- you are deciding how to implement something,
- you want a scoped plan,
- you want file-by-file changes before coding.

Prompt shape:

Read `docs/project_overview.md`, `docs/mvp_scope.md`, `docs/dataset_pamap2.md`, `docs/mvp_build_recipe.md`, and the current codebase.

Create a practical step-by-step plan for: [insert task].

Requirements:
- do not write code yet
- list assumptions first
- list files to edit or create
- identify leakage risks
- keep the solution within current MVP scope
- define what done looks like

### 2. Implementation
Use this when:
- you already know the task,
- you want code edits,
- you want Copilot to help write notebook or Python code.

Prompt shape:

Read `docs/project_overview.md`, `docs/mvp_scope.md`, `docs/dataset_pamap2.md`, `docs/mvp_build_recipe.md`, and the current codebase.

Implement only this task: [insert task].

Before writing code:
- summarize the plan in 5 to 10 bullets
- list exact files to modify
- state assumptions clearly

After writing code:
- summarize what changed
- list anything unfinished
- tell me the next smallest step

### 3. General questions
Use this when:
- you want an explanation,
- you want tradeoffs,
- you want debugging advice,
- you want help interpreting results.

Prompt shape:

Read `docs/project_overview.md`, `docs/mvp_scope.md`, `docs/dataset_pamap2.md`, and the relevant notebook or file.

Answer this question in the context of the current MVP:
[insert question]

Be practical. Point out risks, weak assumptions, and likely failure modes.

---

## Session hygiene

Use one chat session per task.

Good pattern:
- one session for planning the next step
- one session for implementing that step
- a separate session for debugging or interpretation

Do not keep one giant forever-chat. Start fresh when the topic changes.

---

## How to keep Copilot on scope

At the start of any meaningful session, make Copilot read the docs.

A good short opening message is:

Read `docs/project_overview.md`, `docs/mvp_scope.md`, `docs/dataset_pamap2.md`, and `docs/mvp_build_recipe.md` before answering.

This avoids most drift.

---

## Language and reasoning standard

For this project, ask Copilot to follow these writing rules in every response and generated artifact:

- use simple professional language,
- keep wording understandable at high-school reading level,
- avoid unnecessary jargon,
- include the reason (why) for each high-level decision,
- mention key tradeoffs whenever a high-level choice is made.

Apply this standard to:
- notebook markdown,
- documentation,
- code comments,
- chat summaries,
- commit and PR text.

Useful one-line reminder to paste into prompts:

Use simple professional language at high-school reading level, and include the why and key tradeoffs for every high-level decision.

---

## Rules for asking coding questions

When the task is about code, always specify:
- which file or notebook you are working in,
- what the exact subtask is,
- whether you want planning, implementation, or explanation,
- whether code should be minimal or more polished.

Example:

Read `docs/project_overview.md`, `docs/mvp_scope.md`, `docs/dataset_pamap2.md`, and `docs/mvp_build_recipe.md`.

I am working in `notebooks/01_ingest_and_audit.ipynb`.

Help me implement only the raw data loading and initial audit for one subject. Keep it simple and readable. Do not move on to feature engineering yet.

---

## Rules for asking planning questions

When you want planning help, force Copilot to stay concrete.

Good planning prompt:

Read the project docs first.

Plan only Phase 2 from `docs/mvp_build_recipe.md` for this codebase.
Do not write code yet.
I want:
- exact transformations
- expected intermediate outputs
- functions or cells to create
- edge cases to handle
- what to save to disk

---

## Rules for asking debugging questions

When debugging:
- include the error,
- include the relevant code cell or file,
- include what you expected to happen,
- include whether the problem is logic, shape mismatch, time handling, or leakage risk.

Good debugging prompt:

Read the project docs first.

I am debugging `notebooks/02_feature_pipeline.ipynb`.
Here is the code and the error.
Tell me:
1. the likely cause,
2. the minimum fix,
3. whether this creates leakage or invalid targets,
4. how to test the fix.

---

## Rules for asking interpretation questions

When you already have results, ask Copilot to interpret them in project context.

Good interpretation prompt:

Read the project docs first.

Here are my regression metrics, classification metrics, and plots.
Interpret these results for this project's actual goals.
Tell me:
- what is genuinely good,
- what is weak,
- what I can honestly claim,
- what the highest-ROI next improvement would be.

---

## Required behavior when Copilot proposes something big

If Copilot proposes:
- deep learning,
- Dask everywhere,
- more infrastructure,
- deployment,
- cloud services,
- many extra models,

make it justify the change against current scope.

Use this prompt:

Stay within the current MVP unless you can justify a scope change.
Explain:
1. why this is necessary,
2. why it is worth the time,
3. what simpler option you rejected,
4. what new resume signal it adds.

---

## Best first prompt after setting up the repo

Read `docs/project_overview.md`, `docs/mvp_scope.md`, `docs/dataset_pamap2.md`, and `docs/mvp_build_recipe.md`.

I am at the start of the project.
Help me complete only Phase 1 from the build recipe inside `notebooks/01_ingest_and_audit.ipynb`.
Do not move on to later phases.
Before giving code, summarize the exact cells I should create and what each one should do.

---

## Best prompt when moving to the next phase

Read the project docs first.

I finished Phase 1 and updated `docs/dataset_pamap2.md` with what I found.
Now help me implement only Phase 2 and Phase 3 in `notebooks/02_feature_pipeline.ipynb`.
Keep the code simple, readable, and well commented.
Guard carefully against subject leakage and future leakage.

---

## Best prompt when writing final project language

Read the project docs first.

Based on the current notebooks and results, draft:
1. a truthful project summary,
2. 2 to 4 resume bullet options,
3. a brief interview explanation of the modeling choices.

Do not overstate anything that is not already implemented.

---

## Golden rule

The docs are the memory.
The notebooks are the build.
Each Copilot session should be scoped to one task.