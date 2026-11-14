# Credit Card Fraud Detection

This project explores how to identify fraudulent credit-card transactions while making the trade-off between missed fraud and false alerts explicit. It uses a public historical dataset as a learning and benchmarking exercise; it is not a production-ready payment fraud system.

## Business framing

A fraud detection system supports a decision about a transaction, such as approving it, declining it, requesting additional verification, or sending it for review. Those actions have different consequences:

| Outcome | Potential business impact |
| --- | --- |
| Fraud is not detected (false negative) | Fraud losses and related investigation or dispute costs may increase. |
| Legitimate transaction is flagged (false positive) | A good customer may face friction or a declined purchase, harming trust and revenue. |
| Fraud is detected (true positive) | The transaction can be handled according to the institution's response policy. |

The actual cost of these outcomes depends on the issuer, payment flow, recovery rate, review capacity, and customer impact. This dataset does not provide enough information to calculate a real institution's full cost matrix.

Fraud is rare relative to legitimate activity. Consequently, accuracy alone is not a useful success criterion: a classifier that labels every transaction legitimate can achieve high accuracy while detecting no fraud. Evaluation should include Precision-Recall AUC (PR-AUC), precision, recall, false-alert volume, and threshold-specific results. F2-score may be reported when the project explicitly prioritizes recall, but it is not a substitute for a business-approved operating threshold. Any monetary cost analysis must state its assumptions.

Production systems also have operational constraints. Decision latency, review capacity, fallback behavior, privacy, and monitoring requirements must be defined by the system that will use the model. This dataset can support model experiments, but cannot validate an end-to-end production service-level objective.

## Dataset

**Source:** [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The Kaggle description reports 284,807 transactions from two days in September 2013, with 492 fraud labels (about 0.172%). Each row represents a transaction. The target column is `Class` (`1` for fraud and `0` for legitimate).

| Column(s) | Meaning and limitation |
| --- | --- |
| `Time` | Seconds elapsed since the first transaction in the dataset; not a calendar timestamp. |
| `Amount` | Transaction amount. It may be used as an amount-related signal, but does not represent the full business cost of a fraud. |
| `V1`–`V28` | Anonymized PCA-transformed input variables. Their original business meanings are not provided. |
| `Class` | Binary fraud label. |

## What this dataset supports

- Inspecting the schema, data types, missing values, duplicates, label distribution, and suspicious values.
- Exploring `Time`, `Amount`, and the relationships between the anonymized PCA variables and `Class`.
- Training and comparing binary classifiers, including a neural network and simpler baselines.
- Measuring performance under severe class imbalance with PR-AUC and threshold-specific precision, recall, and false-alert counts.
- Comparing thresholds to show how detecting more fraud can increase alerts on legitimate transactions.
- Benchmarking model-only inference time in a stated local environment.
- Using a chronological evaluation split as a limited simulation of scoring later transactions, after verifying the ordering and documenting the dataset's short time span.

## What this dataset does not support

- Per-card or per-customer histories, transaction velocity windows, or customer-specific spending baselines: no card or customer identifiers are provided.
- Merchant-specific risk, device fingerprints, IP or VPN signals, geographic distance, or impossible-travel checks: those fields are absent.
- Calendar-time analysis such as hour of day, weekday, holidays, or seasonality: `Time` is relative elapsed seconds, not an absolute timestamp.
- Measuring actual approval, decline, step-up verification, chargeback, recovery, investigation, or customer-churn outcomes: those business-process labels are absent.
- Deriving the institution's true false-positive and false-negative costs or claiming a financially optimal threshold. `Amount` alone is only a partial proxy.
- Verifying an end-to-end payment authorization SLA or production memory and infrastructure requirements. A local model benchmark is not a production service benchmark.
- Demonstrating current fraud performance, robust concept-drift handling, or reliable generalization to other periods, regions, issuers, or transaction channels. The data covers only two days from 2013.
- Fairness across demographic groups, because demographic attributes are not included.

The dataset therefore supports a historical benchmark, not claims that a model is ready to protect a current financial institution. Results must be reported with these limitations.

## Learning workflow

The project uses three Marimo applications, kept as ordinary Python files for review and version control:

1. [01 — Dataset understanding and EDA](notebooks/01_dataset_understanding_eda.py) audits the pinned dataset and explores target prevalence, amount and PCA distributions, class associations, correlation, relative elapsed time, and outliers.
2. [02 — Data preparation and baselines](notebooks/02_data_preparation_baselines.py) creates grouped outer and inner holdouts, explains train-only preprocessing, and offers a button-triggered baseline comparison.
3. [03 — MLP grouped cross-validation and final evaluation](notebooks/03_mlp_training_evaluation.py) gates a two-candidate, three-fold grouped search and one final evaluation on the reserved outer test behind an explicit run button.

The detailed [EDA report](reports/eda_summary.md) preserves the collected diagnostic findings. The [baseline validation report](reports/baseline_validation.md) preserves the exact tree-model validation metrics and confusion matrices. The MLP grouped cross-validation, training/refit on development data, and one final evaluation of the reserved outer test are complete; the exact candidate scores, selected parameters, test metrics, and confusion matrix are in the [MLP evaluation report](reports/mlp_final_evaluation.md). The test was evaluated once after selection, with no threshold tuning.

KaggleHub retrieves dataset handle `mlg-ulb/creditcardfraud/versions/3` only when `data/raw/creditcard.csv` is absent, and copies only that CSV into `data/raw/`. The raw data directory is ignored by Git; the CSV must not be committed. Check Kaggle's usage terms before redistributing the data. KaggleHub may require account setup depending on your environment; do not store credentials in this repository.

## Setup and run

From the repository root, create an environment and install the project and development tools:

Python 3.12 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[dev]'
```

Run the lightweight tests and open an app in the Marimo editor:

```bash
python3 -m pytest
marimo edit notebooks/01_dataset_understanding_eda.py
marimo edit notebooks/02_data_preparation_baselines.py
marimo edit notebooks/03_mlp_training_evaluation.py
```

Notebook 01 downloads the pinned CSV if needed and performs descriptive analysis. Notebook 02 loads the local data on open, then gates grouped split preparation and baseline fitting behind buttons; its three baselines train only on the inner training partition. Notebook 03 does not load data or run a search until **Run bounded MLP search and final test** is clicked. It uses the combined development partitions for grouped CV, refits the selected pipeline on development, then evaluates the reserved outer test once. Use `make eda`, `make baselines`, and `make test` as shortcuts.

## Project map

```text
docs/                 Business framing, data, architecture, methodology, model card
data/raw/             Local Kaggle CSV (ignored); .gitkeep tracks the empty directory
notebooks/            Three Marimo Python applications
reports/              EDA, baseline validation, and completed MLP evaluation evidence
src/fraud_detection/  Data validation, grouped splits, pipelines, plots, metrics
tests/                Lightweight synthetic-data tests; no downloads or model fitting
```

## Data quality and evaluation principles

Preserve the downloaded source file and document its provenance. Audit the expected columns and types, missingness, exact duplicates, label values, feature ranges, and relative ordering. Do not silently delete, cap, or rewrite anomalous rows during this descriptive audit. `Time` is elapsed seconds rather than calendar time, and `V1`–`V28` are anonymized PCA inputs without direct business meaning.

The current evaluation holds identical input-feature groups together in the outer and inner split. Baseline preprocessing is fitted only on the inner training subset; validation and test retain natural prevalence. The MLP search and development-data refit are complete: only inner training and validation data were combined for three-fold `StratifiedGroupKFold` search with group IDs, with scaling and SMOTE inside each fold's training pipeline. The selected pipeline was refit on development data before the reserved test was evaluated once. The test contains 56,962 rows and 99 frauds; it was not used for search or threshold tuning. Tree-baseline results are validation-only and are not directly comparable with MLP outer-test results because the partitions differ. The 99-fraud holdout estimate is uncertain and cannot establish current or production performance for this two-day 2013 dataset. Do not use the outer test again for iteration.

## Documentation

- [Business requirements](docs/business_requirements.md)
- [Data dictionary](docs/data_dictionary.md)
- [Architecture](docs/architecture.md)
- [Methodology](docs/methodology.md)
- [Model card](docs/model_card.md)
