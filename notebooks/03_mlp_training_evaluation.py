import marimo

__generated_with = "0.23.16"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from fraud_detection.data import FEATURE_COLUMNS, load_dataset
    from fraud_detection.mlp import evaluate_mlp_test, run_mlp_search
    from fraud_detection.splitting import create_grouped_partitions

    return (
        FEATURE_COLUMNS,
        create_grouped_partitions,
        evaluate_mlp_test,
        load_dataset,
        mo,
        pd,
        run_mlp_search,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 03 — MLP grouped cross-validation and final evaluation

    The documented MLP cross-validation, development-data training/refit, and one final reserved-test evaluation are complete. Opening this app does not load the CSV, start a search, fit a model, or score the reserved test. The run button below would repeat the search and score the reserved test again; do not use it for further iteration. Review the recorded results and evaluation boundary below.

    ## Why an MLP, scaling, and SMOTE?

    The 30 numeric inputs have very different scales, while an MLP is sensitive to feature scale. The pipeline therefore applies **`StandardScaler -> SMOTE(random_state=42) -> MLPClassifier`**. The imbalanced-learn pipeline is inside cross-validation, so scaling and resampling are fitted only on each fold's training data; each validation fold stays at natural prevalence. Early stopping is disabled because its internal validation split would be taken from the post-SMOTE training data.

    ## Bounded grouped search

    The inner model-training and validation partitions are combined as development data; the reserved outer test is not included. A `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=44)` search receives the development group IDs, keeping exact-feature duplicates together. Three folds provide a practical initial estimate with 393 development frauds; they do not make the result exhaustive.

    The two candidates differ in network size and L2 regularization:

    | Candidate | Hidden layers | `alpha` | `learning_rate_init` |
    | --- | --- | ---: | ---: |
    | 1 | `(32,)` | `1e-3` | `1e-3` |
    | 2 | `(64, 32)` | `1e-4` | `1e-3` |

    Both use ReLU, Adam, batch size 256, `max_iter=100`, `early_stopping=False`, and seed 42. Search selection uses mean fold Average Precision. This is a **bounded initial search**, not an exhaustive optimum.

    ## Final-test boundary

    After cross-validation, `GridSearchCV(refit=True)` fits the selected pipeline on all development rows. The outer test is then scored once and never used to select parameters or tune a threshold. Average Precision uses `predict_proba`; precision, recall, F2, and confusion counts use the estimator's default `predict` labels. The test contains 56,962 rows and only 99 frauds, so its metrics are one final holdout estimate, not proof of production performance.
    """)
    return


@app.cell
def _(mo):
    mlp_run_button = mo.ui.run_button(label="Run bounded MLP search and final test")
    mlp_run_button
    return (mlp_run_button,)


@app.cell
def _(
    FEATURE_COLUMNS,
    create_grouped_partitions,
    evaluate_mlp_test,
    load_dataset,
    mlp_run_button,
    mo,
    run_mlp_search,
):
    mo.stop(
        not mlp_run_button.value,
        mo.md(
            "Click the button above to run grouped cross-validation and then "
            "score the reserved test once."
        ),
    )
    dataset = load_dataset()
    features = dataset.loc[:, list(FEATURE_COLUMNS)]
    target = dataset["Class"]
    partitions = create_grouped_partitions(features, target)
    mlp_search_result = run_mlp_search(partitions)
    mlp_test_result = evaluate_mlp_test(
        mlp_search_result.best_estimator,
        partitions.x_test,
        partitions.y_test,
    )
    return mlp_search_result, mlp_test_result


@app.cell
def _(mo):
    mo.md(r"""
    ## Search results
    """)
    return


@app.cell
def _(mlp_search_result, pd):
    candidate_cv_results = pd.DataFrame(
        [
            {
                "candidate": index,
                "selected": candidate.params == mlp_search_result.best_params,
                "hidden_layer_sizes": candidate.params["hidden_layer_sizes"],
                "alpha": candidate.params["alpha"],
                "learning_rate_init": candidate.params["learning_rate_init"],
                "mean_cv_average_precision": candidate.mean_average_precision,
                "std_cv_average_precision": candidate.std_average_precision,
                "fold_count": candidate.fold_count,
            }
            for index, candidate in enumerate(mlp_search_result.candidates, start=1)
        ]
    )
    candidate_cv_results
    return (candidate_cv_results,)


@app.cell
def _(mlp_search_result, pd):
    best_cv_score = pd.DataFrame(
        [
            {
                "best_mean_cv_average_precision": (
                    mlp_search_result.best_average_precision
                )
            }
        ]
    )
    best_cv_score
    return (best_cv_score,)


@app.cell
def _(mlp_search_result, pd):
    best_params = pd.DataFrame(
        [
            {"parameter": name, "selected_value": value}
            for name, value in mlp_search_result.best_params.items()
        ]
    )
    best_params
    return (best_params,)


@app.cell
def _(mo):
    mo.md(r"""
    ## One final reserved-test evaluation
    """)
    return


@app.cell
def _(mlp_test_result, pd):
    final_test_metrics = pd.DataFrame(
        [
            {
                "test_rows": mlp_test_result.row_count,
                "test_frauds": mlp_test_result.fraud_count,
                "average_precision": mlp_test_result.average_precision,
                "precision": mlp_test_result.precision,
                "recall": mlp_test_result.recall,
                "f2": mlp_test_result.f2,
            }
        ]
    )
    final_test_metrics
    return (final_test_metrics,)


@app.cell
def _(mlp_test_result, pd):
    confusion_counts = pd.DataFrame(
        [
            {
                "TN": mlp_test_result.tn,
                "FP": mlp_test_result.fp,
                "FN": mlp_test_result.fn,
                "TP": mlp_test_result.tp,
            }
        ],
        columns=["TN", "FP", "FN", "TP"],
    )
    confusion_counts
    return (confusion_counts,)


@app.cell
def _(mlp_test_result):
    mlp_test_result.preview
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation guide

    The completed three-fold search selected candidate 2, `(64, 32)` with `alpha=0.0001` and `learning_rate_init=0.001`, with mean CV Average Precision `0.7892942255438676` and standard deviation `0.024821041864665654`. Candidate 1, `(32,)` with `alpha=0.001`, scored mean `0.7889909900343617` and standard deviation `0.016266031474164548`. Candidate 2 is the search winner, but its mean advantage is tiny relative to fold variation and does not establish decisive architecture superiority.

    After refitting the selected pipeline on all development data, the reserved test was evaluated once: Average Precision `0.8203794692697799`, precision `0.7155172413793104`, recall `0.8383838383838383`, and F2 `0.810546875` over 56,962 rows with 99 frauds. No threshold tuning occurred; `predict` supplies default decision labels, not a business-approved operating point. This is an uncertain holdout estimate, not evidence of current-fraud or production performance. Do not use the outer test again for iteration. The tree baselines have validation-only results and are not directly comparable because they use a different partition. See the [completed MLP evaluation report](../reports/mlp_final_evaluation.md) for the full results and confusion matrix.
    """)
    return


if __name__ == "__main__":
    app.run()
