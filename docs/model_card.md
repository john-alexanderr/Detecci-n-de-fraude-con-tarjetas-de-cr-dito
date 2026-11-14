# Model Card: Preliminary Fraud Baselines

## Summary

This card describes a historical academic comparison of Decision Tree, Random Forest, and AdaBoost for binary fraud classification. Results are preliminary validation evidence, not a deployed model or an approval to make payment decisions.

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

The baseline results use one duplicate-group-aware inner validation partition with 45,569 rows and 79 frauds. A scaler and SMOTE are fitted inside each training-only pipeline; validation data retain natural prevalence. Classifiers use their default thresholds. The outer test set has not been scored. Exact metrics and confusion matrices are in [the validation report](../reports/baseline_validation.md).

Random Forest has the strongest provisional balance in this one split. AdaBoost has the highest recall but also a large false-alert burden. These findings are not estimates of performance outside this partition.

## Limitations and risks

- Severe class imbalance and only 79 fraud examples in validation make estimates uncertain.
- The dataset covers two days in 2013; current and temporal generalization are unknown.
- Thresholds have not been selected against an institution-approved cost function or review capacity.
- The data do not support demographic fairness analysis or card/customer-level behavior.
- No cross-validation, tuning, or final outer-test evaluation exists.

## Untrained hypothesis

A proposed MLP uses 30 numeric inputs, hidden layers `(64, 32)`, ReLU activation, binary log-loss, Adam with `learning_rate_init=1e-3`, `alpha=1e-4`, batch size 256, `max_iter=100`, and seed 42. This configuration is **not trained or validated** and is not included in the reported model comparison. Do not enable early stopping inside a post-SMOTE estimator: it would split resampled data rather than provide the intended natural-prevalence validation.
