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

## Download

The dataset can be downloaded with [KaggleHub](https://github.com/Kaggle/kagglehub):

```python
import kagglehub

# Download the latest version.
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)
```

This call downloads the latest version and returns its local path (which may be KaggleHub's cache). For reproducible results, record and use a specific dataset version when one is selected, along with the retrieval date and a checksum. Check Kaggle's usage terms before redistributing the data. The raw dataset should not be committed to this repository; commit the code and documentation needed to retrieve and audit it instead.

## Data quality and evaluation principles

Before preprocessing or modeling, preserve the downloaded source file and document its provenance. Audit its shape, expected columns and types, missingness, exact duplicates, label values and class counts, feature ranges, and temporal ordering. Do not silently remove or rewrite anomalous rows during the initial audit.

For evaluation, keep validation and test data at the natural class prevalence. Any resampling or learned preprocessing must be fitted only on the training partition. A chronological split is preferable for a future-facing experiment, but two days of data provide weak evidence about temporal generalization. Report that limitation rather than presenting the split as production validation.

## Current scope

1. Establish the business framing and the dataset's supported claims and limitations.
2. Download and audit the dataset, recording its provenance and quality findings.
3. Define an evaluation protocol before model comparison.
