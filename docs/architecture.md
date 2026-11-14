# Project Architecture

## Overview

The repository is a small academic benchmark, organized around two Marimo applications and reusable analysis modules. It intentionally contains no empty service, API, deployment, monitoring, or streaming layers.

```text
docs/                 Project framing and methodology
data/raw/             Ignored local dataset copy
notebooks/            Marimo EDA and grouped-baseline applications
reports/              Preserved analysis and validation results
src/fraud_detection/  Dataset, split, model, plot, and metric helpers
tests/                Synthetic-data unit tests
```

## Responsibilities

| Component | Responsibility |
| --- | --- |
| `fraud_detection.data` | Resolve the repository-local data path, retrieve the pinned source only when absent, and validate its schema. |
| `fraud_detection.splitting` | Hash input-feature signatures and create grouped outer and inner stratified holdouts. |
| `fraud_detection.models` | Construct the three baseline pipelines in leak-safe preprocessing order. |
| `fraud_detection.plots` | Provide reusable descriptive visualizations for the EDA application. |
| `fraud_detection.evaluation` | Fit and evaluate baselines on training and validation data only. |
| `notebooks/01_dataset_understanding_eda.py` | Load and describe the source, then render the documented EDA views. |
| `notebooks/02_data_preparation_baselines.py` | Show grouped partitions and run model fitting only after an explicit button action. |

## Data flow and boundaries

1. `data.raw.creditcard.csv` is reused if present; otherwise KaggleHub retrieves dataset version 3 and only `creditcard.csv` is copied locally.
2. Schema validation checks the expected columns, numeric inputs, and binary target without rewriting records.
3. Outer and inner group-aware splits use hashes of `X` only; target values do not define group membership.
4. During the optional baseline action, the complete `StandardScaler -> SMOTE -> classifier` pipeline is fit on the training subset. Validation is scored at natural prevalence.
5. The evaluation helper accepts no outer-test data. That partition remains unused.

## Dependency and test configuration

`pyproject.toml` is the single dependency and package configuration source. The pinned analysis dependencies are project requirements; Marimo and pytest are in the `dev` extra. Tests use small synthetic data and do not download the Kaggle file or fit the baseline models.
