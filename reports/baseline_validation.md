# Baseline Validation Results

## Evaluation boundary

These preserved results come from one grouped inner validation partition containing **45,569 rows and 79 frauds**. Validation retains natural class prevalence. The pipelines apply `StandardScaler -> SMOTE -> classifier` to training data only. Precision, recall, F2, and confusion matrices use each classifier's default prediction threshold; Average Precision uses predicted probabilities.

The outer test partition was not used for selection or scoring. Cross-validation, threshold tuning, hyperparameter tuning, and MLP training have not been performed.

## Metrics

| Classifier | Average Precision | Precision | Recall | F2 |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.273289 | 0.359281 | 0.759494 | 0.621118 |
| Random Forest | 0.849946 | 0.876712 | 0.810127 | 0.822622 |
| AdaBoost | 0.736748 | 0.048161 | 0.911392 | 0.198785 |

## Confusion matrices

Rows are true labels and columns are predictions, ordered `[0, 1]`.

| Classifier | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 45,383 | 107 | 19 | 60 |
| Random Forest | 45,481 | 9 | 15 | 64 |
| AdaBoost | 44,067 | 1,423 | 7 | 72 |

## Interpretation

Random Forest is the strongest **provisional** balance on this single validation split: it has the highest Average Precision, precision, and F2 among these three default-threshold results. AdaBoost detects the most frauds (72 of 79; recall 0.911392) but produces 1,423 false alerts and precision 0.048161. The Decision Tree is a reference point between the two behaviors.

This comparison does not establish a final model, business-approved threshold, current fraud performance, or production readiness. The small validation fraud count and two-day 2013 source period limit generalization claims.
