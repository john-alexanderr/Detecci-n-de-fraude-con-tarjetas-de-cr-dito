# Model Card: Preliminary Fraud Baselines and Completed MLP Evaluation

## Summary

This card describes a historical academic comparison of Decision Tree, Random Forest, and AdaBoost, plus completed MLP cross-validation, development-data training/refit, and one final outer-test evaluation. The tree results are validation-only evidence, and no model is approved for payment decisions.

## Intended use

- Teaching and reproducible comparison of classifiers on a public benchmark.
- Studying precision-recall trade-offs under severe class imbalance.

## Out-of-scope use

Do not use these models or results to approve, decline, or otherwise make real payment decisions. They have not been evaluated for current fraud patterns, a specific issuer, operational latency, fairness, review capacity, or production performance.

## Data

The input is the Kaggle credit-card fraud dataset (version 3), reported as 284,807 transactions over two days in 2013, with 492 fraud labels. Inputs are `Time`, `Amount`, and anonymized PCA components `V1`–`V28`; `Class` is the target. There are no card, customer, merchant, demographic, or behavioral-sequence fields. `Time` is elapsed seconds, not calendar time.

## EDA and selection boundary

The full-source inventory in the EDA app is limited to input-integrity diagnostics and label prevalence for split stratification. The preserved [EDA report](../reports/eda_summary.md) contains exact values from a full-source descriptive pre-model audit; those historical results were not used to select features, transformations, models, or thresholds. After an explicit run-button action, the current app computes model-relevant plots and interpretations only from the grouped development partition (outer training plus validation). It does not pass outer-test rows to EDA functions or report outer-test counts.

## Evaluation

The baseline results use one duplicate-group-aware inner validation partition with 45,569 rows and 79 frauds. A scaler and SMOTE are fitted inside each training-only pipeline; validation data retain natural prevalence. Classifiers use their default thresholds. Exact metrics and confusion matrices are in [the validation report](../reports/baseline_validation.md).

Random Forest has the strongest provisional balance in this one split. AdaBoost has the highest recall but also a large false-alert burden. These findings are not estimates of performance outside this partition.

The completed MLP phase combines only inner training and validation data for a three-fold `StratifiedGroupKFold` search, passing the feature-group IDs. Its `StandardScaler -> SMOTE -> MLPClassifier` pipeline keeps scaling and sampling inside each fold. Candidate 2, `hidden_layer_sizes=(64, 32)`, `alpha=0.0001`, and `learning_rate_init=0.001`, was the search winner; its mean CV Average Precision was `0.7892942255438676` (standard deviation `0.024821041864665654`). The mean advantage over candidate 1 is tiny relative to fold variation and does not establish decisive architecture superiority. The selected pipeline was refit on all development data, then the reserved outer test was evaluated once after selection. Its metrics were Average Precision `0.8203794692697799`, precision `0.7155172413793104`, recall `0.8383838383838383`, and F2 `0.810546875`, on 56,962 rows including 99 frauds. Average Precision uses probabilities; precision, recall, F2, and confusion counts use default predictions. No threshold tuning occurred. See the [MLP report](../reports/mlp_final_evaluation.md) for both candidate results and the confusion matrix.

The baseline tree models have validation-only results and are **not directly comparable** with the MLP outer-test metrics because they were evaluated on different partitions.

## Limitations and risks

- Severe class imbalance and only 79 fraud examples in validation make estimates uncertain.
- The final MLP holdout estimate includes only 99 frauds and is uncertain; it must not be used for further iteration or threshold selection.
- The dataset covers two days in 2013; current and temporal generalization are unknown.
- Thresholds have not been selected against an institution-approved cost function or review capacity.
- The data do not support demographic fairness analysis or card/customer-level behavior.
- The MLP search is only a two-candidate initial grid; even after execution, it will not be an exhaustive optimum.
- Neither model phase establishes production readiness or current fraud performance. The MLP result does not imply performance on current fraud patterns or beyond this two-day 2013 dataset.
- Do not use the reserved outer test again for iteration. If desired, cross-validate the tree baselines and design a fair model comparison using development data; select any threshold on development or validation data only.

## Completed MLP search

A bounded search compared two MLP candidates using 30 numeric inputs. Both used ReLU, Adam, `learning_rate_init=0.001`, batch size 256, `max_iter=100`, `early_stopping=False`, and seed 42. The selected search winner used hidden layers `(64, 32)` and `alpha=0.0001`; the selected pipeline was refit on all development data before the single outer-test evaluation. Early stopping was disabled inside the post-SMOTE pipeline so the explicit grouped folds remained the validation boundary. The two-candidate search is not exhaustive, and its winner is not proof of architecture superiority.
