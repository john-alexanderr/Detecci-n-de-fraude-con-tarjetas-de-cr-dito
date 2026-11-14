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

## MLP bounded search and final evaluation

The MLP phase is implemented in [Notebook 03](../notebooks/03_mlp_training_evaluation.py) and shares the existing deterministic grouped partitions. It combines only `x_fit` plus `x_validation`, their targets, and their feature-group IDs into development data. The 56,962-row outer test remains reserved; it is never included in cross-validation.

The search uses `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=44)` and passes development group IDs to `GridSearchCV.fit`. Three folds are a bounded initial choice for the 393 development frauds, not an exhaustive search or an optimality claim. The two candidates are:

| Candidate | `hidden_layer_sizes` | `alpha` | `learning_rate_init` |
| --- | --- | ---: | ---: |
| 1 | `(32,)` | `1e-3` | `1e-3` |
| 2 | `(64, 32)` | `1e-4` | `1e-3` |

Both candidates use ReLU, Adam, batch size 256, `max_iter=100`, `early_stopping=False`, and random state 42. The estimator order is **`StandardScaler -> SMOTE(random_state=42) -> MLPClassifier`** inside the cross-validation estimator. Therefore each fold fits scaling and SMOTE only on its training partition; its validation data keeps natural prevalence. Search uses `average_precision`, `refit=True`, one job, one pre-dispatched job, and `error_score="raise"` so estimator errors are visible. Mean and standard deviation of candidate fold Average Precision are reported with the fold count.

After selection, `GridSearchCV(refit=True)` fits the chosen pipeline on all development rows. The reserved outer test was evaluated once after selection: Average Precision uses `predict_proba`, while precision, recall, F2, and TN/FP/FN/TP confusion counts use default `predict` labels. No threshold tuning occurred. The test has 56,962 rows and 99 frauds, so its result is one uncertain holdout estimate with limited minority-class evidence—not production performance.

| Candidate | Mean CV Average Precision | Std. CV Average Precision | Folds |
| --- | ---: | ---: | ---: |
| 1: `(32,)`, `alpha=0.001`, `learning_rate_init=0.001` | `0.7889909900343617` | `0.016266031474164548` | 3 |
| 2: `(64, 32)`, `alpha=0.0001`, `learning_rate_init=0.001` | `0.7892942255438676` | `0.024821041864665654` | 3 |

Candidate 2 was the search winner and was refit on all development data. Its mean advantage is tiny relative to fold variation, so the result does not demonstrate decisive architecture superiority. The final outer-test results were Average Precision `0.8203794692697799`, precision `0.7155172413793104`, recall `0.8383838383838383`, and F2 `0.810546875`; confusion counts were TN `56830`, FP `33`, FN `16`, and TP `83`. The test was evaluated once, and no threshold tuning occurred. The complete results are in the [MLP result report](../reports/mlp_final_evaluation.md).

The tree baselines have validation-only results and are **not directly comparable** with the MLP outer-test metrics because the evaluated partitions differ. The MLP test estimate is uncertain because it includes only 99 frauds; it establishes neither current-fraud performance nor performance beyond this two-day 2013 dataset. Do not use the reserved outer test again for iteration.

## Limits and future work

The preserved baseline comparison is one validation split with 79 fraud examples and remains provisional. The completed MLP experiment is a bounded initial search, not an exhaustive optimum; its final estimate is based on only 99 outer-test frauds, and the underlying data covers two days in 2013. Neither baseline nor MLP results establish current fraud performance, a business-approved threshold, or production readiness. Future work may cross-validate the tree baselines and design a fair comparison; any threshold selection must use development or validation data only.
