# MLP Grouped Cross-Validation and Final Evaluation

The bounded MLP search, development-data refit, and one final evaluation of the reserved outer test are complete. The test was evaluated once after candidate selection; it was not used for tuning or iteration.

## Evaluation boundary

The inner training and validation partitions were combined as development data. The outer test was excluded from search. A three-fold `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=44)` search received development feature-group IDs. `GridSearchCV` selected by Average Precision and, with `refit=True`, fitted the winning pipeline on all development rows before the reserved test was evaluated once.

The pipeline was `StandardScaler -> SMOTE(random_state=42) -> MLPClassifier`, with scaling and SMOTE fitted only within each CV training fold. Both candidates used ReLU, Adam, batch size 256, `max_iter=100`, `early_stopping=False`, and random state 42.

## Cross-validation results

| Candidate | `hidden_layer_sizes` | `alpha` | `learning_rate_init` | Mean CV Average Precision | Std. CV Average Precision | Folds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `(32,)` | `0.001` | `0.001` | `0.7889909900343617` | `0.016266031474164548` | 3 |
| 2 | `(64, 32)` | `0.0001` | `0.001` | `0.7892942255438676` | `0.024821041864665654` | 3 |

**Selected search winner:** `hidden_layer_sizes=(64, 32)`, `alpha=0.0001`, `learning_rate_init=0.001`.

Candidate 2 has the higher mean CV Average Precision, but its advantage is tiny relative to fold variation. It is the search winner, not decisive evidence that its architecture is superior.

## Final reserved-test evaluation

| Test rows | Test frauds | Average Precision | Precision | Recall | F2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 56962 | 99 | 0.8203794692697799 | 0.7155172413793104 | 0.8383838383838383 | 0.810546875 |

The test was evaluated once after selection. Average Precision uses `predict_proba`; precision, recall, F2, and confusion counts use default `predict` labels. No threshold tuning occurred.

Confusion matrix, with true labels as rows and predicted labels `[0, 1]` as columns:

| True \ Predicted | 0 | 1 |
| --- | ---: | ---: |
| 0 | 56830 | 33 |
| 1 | 16 | 83 |

These are counts in the order TN `56830`, FP `33`, FN `16`, TP `83`.

## Interpretation and limitations

- The CV scores are development-data estimates from a bounded two-candidate search, not an exhaustive optimum.
- The outer-test result is one uncertain holdout estimate based on only 99 frauds. It does not establish current-fraud performance, production readiness, or performance beyond this two-day 2013 dataset.
- The baseline tree models have validation-only results. They are **not directly comparable** with the MLP outer-test metrics because the evaluated partitions differ.
- The notebook generated a prediction preview, but individual prediction rows are not reproduced in static documentation.

## Future work

- Cross-validate and tune the tree baselines if desired.
- Select any operating threshold using development or validation data only.
- Design a fair comparison that evaluates models on the same appropriate partitions and protocol.
- Do not use the reserved outer test again for iteration.
