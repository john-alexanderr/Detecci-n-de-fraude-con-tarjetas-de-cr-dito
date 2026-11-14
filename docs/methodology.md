# Methodology

## Dataset audit and exploratory analysis

The EDA application keeps the full downloaded source intact. Its full-source inventory cells report schema, sample rows, summary statistics, null counts, exact duplicates, and label prevalence for input-integrity review and split stratification only; these are not model-selection evidence. The grouped split is created only after an explicit button click. The app then combines the grouped model-training and validation rows into a development frame, and passes only that frame to class-conditioned plots, target associations, correlations, outlier fences, and other model-relevant EDA. Outer-test rows are not passed to those functions, and the EDA app does not report outer-test counts.

The preserved [EDA report](../reports/eda_summary.md) remains an exact full-source descriptive pre-model audit, including its previously recorded feature-level statistics and associations. Those historical values are retained for provenance, not used to fit transformations, select features, choose models or thresholds, or estimate performance. Plots are descriptive, not causal evidence and not a basis for automatic deletion or capping. `Time` is relative elapsed seconds, and `V1`–`V28` have no direct business interpretation.

## Group-aware partitions

1. Hash each row's input features only; do not include `Class` in the group key.
2. Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` and its first fold for the outer development/test partition.
3. Use a second `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)` and its first fold within development for model training and validation.
4. Keep exact input-feature groups entirely within one partition at both levels. Keep all rows; do not deduplicate.

The current outer split has 227,845 development rows (393 frauds) and 56,962 outer-test rows (99 frauds). The inner split has 182,276 model-training rows (314 frauds) and 45,569 validation rows (79 frauds). These splits preserve approximately the natural prevalence and have no cross-partition exact-feature group overlap.

For comparison, the prior conventional stratified random split had 227,845 training rows (394 frauds) and 56,962 test rows (98 frauds), but 293 distinct exact-feature signatures crossed the split. This overlap is why the retained evaluation uses grouped partitions.

## Preprocessing and baselines

Each baseline pipeline applies **`StandardScaler -> SMOTE(random_state=42) -> classifier`**. The scaler and sampler are fitted only on the training subset; generated samples never enter validation. Validation remains at natural prevalence. The three classifiers and settings are:

| Classifier | Settings |
| --- | --- |
| Decision Tree | `random_state=42` |
| Random Forest | `n_estimators=200`, `random_state=42`, `n_jobs=-1` |
| AdaBoost | `random_state=42` |

Metrics use each classifier's default prediction threshold. Average Precision is computed from validation probabilities; precision, recall, F2, and confusion matrices use `predict` output. The exact previous validation results are documented in [Baseline Validation](../reports/baseline_validation.md).

## Limits and pending work

The baseline comparison is one validation split with 79 fraud examples. It is provisional; no cross-validation, threshold tuning, hyperparameter tuning, MLP training, or outer-test scoring has been performed. The proposed MLP is a hypothesis only. A later phase must define group-aware validation, freeze choices before a single outer-test evaluation, and report the dataset's two-day 2013 limitation.
