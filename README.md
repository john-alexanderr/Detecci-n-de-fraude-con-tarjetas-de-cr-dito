# Credit Card Fraud Detection

A reproducible academic benchmark for detecting fraudulent credit-card transactions under severe class imbalance. It is a learning exercise on a public historical dataset, not a production-ready fraud system.

## What was done

1. **Dataset audit and EDA** — verified schema, types, missingness, duplicates, label values, and feature ranges without modifying anomalous rows. Explored target prevalence, `Amount`, the anonymized PCA features `V1`–`V28`, and relative elapsed time. Findings: [reports/eda_summary.md](reports/eda_summary.md).
2. **Leak-safe grouped splits** — identical input-feature rows are kept together in the same partition. Development data (227,845 rows) contains an inner training subset (182,276) and a validation subset (45,569, 79 frauds). The reserved outer test (56,962 rows, 99 frauds) is never used for fitting, selection, or tuning.
3. **Tree baselines** — `StandardScaler -> SMOTE -> classifier` fitted only on the inner training subset; validation keeps natural prevalence. Validation results: [reports/baseline_validation.md](reports/baseline_validation.md).

   | Model | Average Precision | Precision | Recall | F2 |
   | --- | ---: | ---: | ---: | ---: |
   | Decision Tree | 0.2733 | 0.3593 | 0.7595 | 0.6211 |
   | Random Forest | 0.8499 | 0.8767 | 0.8101 | 0.8226 |
   | AdaBoost | 0.7367 | 0.0482 | 0.9114 | 0.1988 |

4. **MLP training, grouped CV, and final test** — development data only for the search: `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=44)` with feature-group IDs and `GridSearchCV` scoring Average Precision. Two candidates, with `StandardScaler -> SMOTE -> MLPClassifier` inside every fold. The winner was refit on all development data, then the reserved test was evaluated **once**.

   | MLP candidate | Mean CV AP | Std. CV AP |
   | --- | ---: | ---: |
   | `(32,)`, `alpha=1e-3`, `learning_rate_init=1e-3` | 0.7890 | 0.0163 |
   | `(64, 32)`, `alpha=1e-4`, `learning_rate_init=1e-3` (selected) | 0.7893 | 0.0248 |

   Final test results: **Average Precision 0.8204**, precision 0.7155, recall 0.8384, F2 0.8105; confusion counts TN 56,830 / FP 33 / FN 16 / TP 83. Full report: [reports/mlp_final_evaluation.md](reports/mlp_final_evaluation.md).

The baseline metrics are validation-only and are not directly comparable with the MLP outer-test metrics because the evaluated partitions differ. The 99-fraud holdout estimate is uncertain and does not establish current or production fraud performance.

## Dataset

**Source:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), pinned by the code to `mlg-ulb/creditcardfraud/versions/3`.

- 284,807 transactions from two days in September 2013; 492 frauds (≈0.172%).
- 30 numeric inputs: `Time` (elapsed seconds), `Amount`, and anonymized PCA features `V1`–`V28`. Target: `Class` (1 = fraud).
- The first notebook run downloads the CSV through KaggleHub to `data/raw/creditcard.csv`. The raw data is Git-ignored and must not be committed. KaggleHub may require a Kaggle account; never store credentials in the repository.

## Requirements

- **Python 3.12 or newer**
- Runtime dependencies (pinned): `kagglehub==1.0.3`, `numpy==2.5.4`, `pandas==3.1.1`, `matplotlib==3.11.2`, `seaborn==0.13.2`, `scikit-learn==1.9.1`, `imbalanced-learn==0.14.2`
- Development tools: `marimo==0.23.16`, `pytest>=8,<10`

Everything is declared in [pyproject.toml](pyproject.toml); the command set is in the [Makefile](Makefile).

## Replication

```bash
git clone git@github.com:john-alexanderr/Detecci-n-de-fraude-con-tarjetas-de-cr-dito.git
cd Detecci-n-de-fraude-con-tarjetas-de-cr-dito

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[dev]'   # or: make install

python3 -m pytest                    # or: make test
```

Open the Marimo apps as ordinary Python applications:

```bash
marimo edit notebooks/01_dataset_understanding_eda.py   # make eda
marimo edit notebooks/02_data_preparation_baselines.py  # make baselines
marimo edit notebooks/03_mlp_training_evaluation.py
```

Notebook 01 downloads the pinned CSV if missing and runs descriptive analysis. Notebook 02 gates split creation and baseline fitting behind buttons. Notebook 03 does nothing until **Run bounded MLP search and final test** is clicked; a full run takes roughly 15 minutes on CPU. Re-running Notebook 03 would evaluate the reserved test again, so do it only for deliberate verification — the test must not be used for iteration.

## Project map

```text
docs/                 Business framing, data, architecture, methodology, model card
data/raw/             Local Kaggle CSV (ignored); .gitkeep tracks the empty directory
notebooks/            Three Marimo apps: EDA, baselines, MLP
reports/              EDA, baseline validation, and MLP evaluation evidence
src/fraud_detection/  Data loading, grouped splits, pipelines, plots, metrics
tests/                Lightweight synthetic-data tests; no downloads or model fitting
```

## Limitations

- The dataset covers only two days of 2013; current or temporal generalization is unknown.
- Severe imbalance (492 frauds total) makes every estimate uncertain, especially the 99-fraud holdout.
- Thresholds were not tuned to any institution's cost function or review capacity; metrics use default thresholds.
- No card/customer identifiers, merchant, device, or calendar-time fields exist, so customer history, velocity, and seasonality analyses are impossible.

## Documentation

- [Business requirements](docs/business_requirements.md)
- [Data dictionary](docs/data_dictionary.md)
- [Architecture](docs/architecture.md)
- [Methodology](docs/methodology.md)
- [Model card](docs/model_card.md)
