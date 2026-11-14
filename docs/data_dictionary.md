# Data Dictionary

## Source and unit of analysis

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), dataset handle `mlg-ulb/creditcardfraud/versions/3`.
- **Unit:** One transaction per row.
- **Task:** Binary classification of `Class`.
- **Reported source scope:** 284,807 transactions from two days in September 2013; 492 fraud labels.
- **Local copy:** `data/raw/creditcard.csv`, downloaded only if absent and ignored by Git.

## Columns

| Column(s) | Role | Meaning and limitations |
| --- | --- | --- |
| `Time` | Numeric input | Seconds elapsed since the first transaction. It is not a calendar timestamp and does not identify clock hour, weekday, or seasonality. |
| `V1`–`V28` | Numeric inputs | Anonymized PCA-transformed features. Their original business meanings are not provided. |
| `Amount` | Numeric input | Transaction amount; it is not a complete measure of fraud loss or business cost. |
| `Class` | Binary target | `0` = legitimate, `1` = fraud. |

There are 30 numeric inputs and one target. The expected source column order is `Time`, `V1`–`V28`, `Amount`, `Class`.

## Known data-quality findings

The preserved profile reports no missing values, 1,081 exact duplicate rows, and zero exact-feature groups with conflicting targets. Candidate sentinel-like values `Time = 999` (2 rows) and `Amount = 999` (7 rows) are unconfirmed; they must not be automatically recoded. IQR flags indicate values to inspect, not records to delete or cap. See the [EDA report](../reports/eda_summary.md) for script-backed detail.
