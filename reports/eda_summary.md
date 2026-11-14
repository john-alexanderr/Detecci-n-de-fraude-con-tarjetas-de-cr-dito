# EDA Report: Credit Card Fraud Dataset

## Executive Summary

- The dataset contains 284,807 rows and 31 columns; retain the 1,081 exact duplicate rows during this descriptive audit, as there are no feature-conflicting-target groups (`eda-profiler.shape={"columns":31,"rows":284807}`; `eda-profiler.duplicates.exact=1081`; `eda-profiler.duplicates.feature_conflicting_targets=0`).
- Fraud is a severe minority class: 492 frauds (0.1727%) versus 284,315 legitimate transactions (99.8273%); prioritize PR-AUC, recall, and F1 and keep any resampling or class weighting inside training folds (`eda-target-analysis.classes=[{"label":"0","count":284315,"pct":99.8273},{"label":"1","count":492,"pct":0.1727}]`; `eda-target-analysis.imbalance=severe`; `eda-target-analysis.recommendations=PR-AUC/recall/F1, stratified CV, train-fold-only resampling/weights`).
- No missing values were detected, but the sentinel-like values are candidates only—not confirmed missing—and IQR flags warrant review rather than automatic row removal or capping (`eda-missingness-audit.rows_with_missing_pct=0.0`; `eda-missingness-audit.sentinel_candidates.Time=999,count=2`; `eda-missingness-audit.sentinel_candidates.Amount=999,count=7`; `eda-missingness-audit.outliers_iqr.Amount=31904`).
- The leakage scan recommends stratified k-fold; drift is unknown without a separate test CSV, and the moderate V14 single-feature AUC is not proof of leakage (`eda-leakage-audit.cv_recommendation="stratified k-fold (StratifiedKFold)"`; `eda-leakage-audit.drift=[]`; `eda-leakage-audit.single_feature_auc.V14.auc=0.9492`).

## Dataset Health

| Phase | Signal | Value | Source | Action |
| --- | --- | --- | --- | --- |
| Structure | Exact duplicates and conflicting targets | 1,081 exact duplicates; zero feature-conflicting-target groups (`eda-profiler.duplicates.exact=1081`; `eda-profiler.duplicates.feature_conflicting_targets=0`) | `eda-profiler.duplicates` | Keep rows; the diagnostics do not support deduplication. |
| Structure | Near-constant target | `Class.quasi_constant=true` (`eda-profiler.Class.quasi_constant=true`) | `eda-profiler.Class.quasi_constant` | Treat as an imbalance signal, not a reason to remove the target. |
| Target | Class imbalance | 492 frauds (0.1727%) and 284,315 legitimate transactions (99.8273%); `imbalance=severe` (`eda-target-analysis.classes=[{"label":"0","count":284315,"pct":99.8273},{"label":"1","count":492,"pct":0.1727}]`; `eda-target-analysis.imbalance=severe`) | `eda-target-analysis.classes`; `eda-target-analysis.imbalance` | Use PR-AUC, recall, and F1; preserve natural prevalence in validation data. |
| Missingness | Nulls and candidate sentinels | Rows with missing values: 0.0%; candidate `Time=999` (2) and `Amount=999` (7) values are unconfirmed (`eda-missingness-audit.rows_with_missing_pct=0.0`; `eda-missingness-audit.sentinel_candidates.Time=999,count=2`; `eda-missingness-audit.sentinel_candidates.Amount=999,count=7`) | `eda-missingness-audit.rows_with_missing_pct`; `eda-missingness-audit.sentinel_candidates` | Confirm sentinel meanings with domain context before treating them as missing. |
| Signals | Skew and target associations | `Amount.skew=16.9776`; strongest listed Pearson associations include `V17=-0.3265` and `V14=-0.3025` (`eda-feature-signals.numeric.Amount.skew=16.9776`; `eda-feature-signals.numeric.V17.pearson_target=-0.3265`; `eda-feature-signals.numeric.V14.pearson_target=-0.3025`) | `eda-feature-signals.numeric` | Treat correlations as associations, not causation; validate candidate signals on partitions. |
| Leakage | Ordering, groups, and drift | `row_index_target_corr=-0.0117`; no group keys or date columns; drift unknown because no separate test CSV was supplied (`eda-leakage-audit.row_index_target_corr=-0.0117`; `eda-leakage-audit.group_keys=[]`; `eda-leakage-audit.temporal.date_columns=[]`; `eda-leakage-audit.drift=[]`) | `eda-leakage-audit` | Keep the grouped holdout documented in Notebook 02; do not claim drift is absent. |

## Findings by Phase

### Structure

- The profile reports 284,807 rows, 31 columns, and an estimated memory footprint of 70,632,136 bytes (`eda-profiler.shape={"columns":31,"rows":284807}`; `eda-profiler.memory_estimate_bytes=70632136`).
- Exact duplicates are 1,081 rows (0.3796%), with zero feature-conflicting-target groups; `Class` is quasi-constant. These findings describe the data and do not authorize deleting records (`eda-profiler.duplicates.exact=1081`; `eda-profiler.duplicates.exact_pct=0.3796`; `eda-profiler.duplicates.feature_conflicting_targets=0`; `eda-profiler.Class.quasi_constant=true`).

### Target

- The class distribution is 284,315 legitimate rows (99.8273%) and 492 fraud rows (0.1727%); the diagnostic class is severe imbalance (`eda-target-analysis.classes=[{"label":"0","count":284315,"pct":99.8273},{"label":"1","count":492,"pct":0.1727}]`; `eda-target-analysis.minority_pct=0.1727`; `eda-target-analysis.imbalance=severe`).
- The supplied recommendations include PR-AUC, recall, and F1; stratified cross-validation; and resampling or weights applied only within training folds (`eda-target-analysis.recommendations=PR-AUC/recall/F1, stratified CV, train-fold-only resampling/weights`).

### Missingness

- No rows contain missing values, all columns have zero missing entries, and no co-missing patterns were found (`eda-missingness-audit.rows_with_missing_pct=0.0`; `eda-missingness-audit.columns[*].missing=0`; `eda-missingness-audit.patterns=[]`).
- Candidate sentinel-like values are `Time=999` in 2 rows and `Amount=999` in 7 rows; they are not confirmed missing values (`eda-missingness-audit.sentinel_candidates.Time=999,count=2`; `eda-missingness-audit.sentinel_candidates.Amount=999,count=7`).
- IQR flag counts include `Amount=31904`, `V27=39163`, `V28=30342`, `V20=27770`, and `V8=24134` (`eda-missingness-audit.outliers_iqr.Amount=31904`; `eda-missingness-audit.outliers_iqr.V27=39163`; `eda-missingness-audit.outliers_iqr.V28=30342`; `eda-missingness-audit.outliers_iqr.V20=27770`; `eda-missingness-audit.outliers_iqr.V8=24134`). These are inspection flags, not evidence of invalid records; do not cap or delete observations during this audit.

### Signals

- `Amount` is strongly right-skewed (`Amount.skew=16.9776`), while `Time` is near symmetric (`Time.skew=-0.0356`) (`eda-feature-signals.numeric.Amount.skew=16.9776`; `eda-feature-signals.numeric.Time.skew=-0.0356`).
- The strongest listed Pearson target associations are `V17=-0.3265`, `V14=-0.3025`, `V12=-0.2606`, and `V10=-0.2169` (`eda-feature-signals.numeric.V17.pearson_target=-0.3265`; `eda-feature-signals.numeric.V14.pearson_target=-0.3025`; `eda-feature-signals.numeric.V12.pearson_target=-0.2606`; `eda-feature-signals.numeric.V10.pearson_target=-0.2169`). Correlation is association, not causation.
- No collinear pairs were flagged (`eda-feature-signals.collinear_pairs=[]`).

### Leakage

- The row-index/target correlation is -0.0117; no group keys or calendar date columns were identified (`eda-leakage-audit.row_index_target_corr=-0.0117`; `eda-leakage-audit.group_keys=[]`; `eda-leakage-audit.temporal.date_columns=[]`).
- The recommended cross-validation strategy is stratified k-fold (`eda-leakage-audit.cv_recommendation="stratified k-fold (StratifiedKFold)"`). Notebook 02 separately preserves its duplicate-group-aware holdout; choose the next CV design with both class balance and duplicate grouping in mind.
- The highest listed single-feature AUC is the moderate V14 value of 0.9492; this alone is not proof of leakage (`eda-leakage-audit.single_feature_auc.V14.auc=0.9492`).
- Drift is unknown, not absent: no separate test CSV was supplied, so the drift result is empty (`eda-leakage-audit.drift=[]`).

## Visualization Guidance

| Chart | Purpose | Finding | Notes |
| --- | --- | --- | --- |
| Class-count plot with a log count axis | Make the minority label visible beside the majority class | `eda-target-analysis.classes=[{"label":"0","count":284315,"pct":99.8273},{"label":"1","count":492,"pct":0.1727}]` | Report PR-AUC and class-specific metrics; accuracy alone is insufficient for this imbalance. |
| `log1p(Amount)` distribution and class-wise comparison | Show the long-tailed amount shape and compare class distributions | `eda-feature-signals.numeric.Amount.skew=16.9776`; `eda-missingness-audit.outliers_iqr.Amount=31904` | Keep the transform descriptive; do not cap/delete records based on the plot. |
| PCA component association and class-wise plots | Inspect anonymized component distributions and candidate target associations | `eda-feature-signals.numeric.V17.pearson_target=-0.3265`; `eda-feature-signals.numeric.V14.pearson_target=-0.3025` | Correlations do not imply causation. |
| Relative elapsed-hour counts and fraud-rate panels | Compare volume, fraud counts, and rate over elapsed time | `eda-leakage-audit.temporal.date_columns=[]` | `Time` is not calendar time; no calendar-hour or weekday conclusions are supported. |

## Open Questions and Next Steps

- [ ] Confirm whether the candidate `Time=999` and `Amount=999` values have domain-specific meanings before treating either as missing (`eda-missingness-audit.sentinel_candidates.Time=999,count=2`; `eda-missingness-audit.sentinel_candidates.Amount=999,count=7`).
- [ ] Design cross-validation that preserves the stratification recommendation while accounting for the duplicate-group structure documented in Notebook 02 (`eda-leakage-audit.cv_recommendation="stratified k-fold (StratifiedKFold)"`; `eda-leakage-audit.group_keys=[]`).
- [ ] In the next modeling phase, compare baselines with cross-validation and tune hyperparameters and thresholds only using training/validation partitions; keep resampling or weighting within each training fold (`eda-target-analysis.recommendations=PR-AUC/recall/F1, stratified CV, train-fold-only resampling/weights`).
- [ ] Obtain an appropriate separate or later-period test dataset before making drift claims; current drift status is unknown (`eda-leakage-audit.drift=[]`).

## Sources

| Skill | Field | Value |
| --- | --- | --- |
| `eda-profiler` | `shape` | `{"columns":31,"rows":284807}` |
| `eda-profiler` | `memory_estimate_bytes` | `70632136` |
| `eda-profiler` | `duplicates.exact`; `duplicates.exact_pct`; `duplicates.feature_conflicting_targets` | `1081`; `0.3796`; `0` |
| `eda-profiler` | `Class.quasi_constant` | `true` |
| `eda-target-analysis` | `classes`; `minority_pct`; `imbalance` | `[{"label":"0","count":284315,"pct":99.8273},{"label":"1","count":492,"pct":0.1727}]`; `0.1727`; `severe` |
| `eda-target-analysis` | `recommendations` | PR-AUC/recall/F1; stratified CV; train-fold-only resampling/weights |
| `eda-missingness-audit` | `rows_with_missing_pct`; `columns[*].missing`; `patterns` | `0.0`; `0`; `[]` |
| `eda-missingness-audit` | `sentinel_candidates.Time`; `sentinel_candidates.Amount` | `value=999,count=2`; `value=999,count=7` |
| `eda-missingness-audit` | `outliers_iqr.Amount`; `outliers_iqr.V27`; `outliers_iqr.V28`; `outliers_iqr.V20`; `outliers_iqr.V8` | `31904`; `39163`; `30342`; `27770`; `24134` |
| `eda-feature-signals` | `numeric.Amount.skew`; `numeric.Time.skew` | `16.9776`; `-0.0356` |
| `eda-feature-signals` | `numeric.V17.pearson_target`; `numeric.V14.pearson_target`; `numeric.V12.pearson_target`; `numeric.V10.pearson_target` | `-0.3265`; `-0.3025`; `-0.2606`; `-0.2169` |
| `eda-feature-signals` | `collinear_pairs` | `[]` |
| `eda-leakage-audit` | `row_index_target_corr`; `group_keys`; `temporal.date_columns`; `cv_recommendation`; `drift` | `-0.0117`; `[]`; `[]`; `stratified k-fold (StratifiedKFold)`; `[]` (drift unknown because no separate test CSV was supplied) |
| `eda-leakage-audit` | `single_feature_auc.V14.auc` | `0.9492` (moderate; not proof of leakage) |
