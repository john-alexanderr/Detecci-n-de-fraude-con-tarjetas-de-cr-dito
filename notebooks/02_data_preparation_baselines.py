import marimo

__generated_with = "0.23.16"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from fraud_detection.data import FEATURE_COLUMNS, load_dataset
    from fraud_detection.evaluation import run_baseline_validation
    from fraud_detection.splitting import (
        create_grouped_partitions,
        grouped_overlap_count,
    )

    return (
        FEATURE_COLUMNS,
        create_grouped_partitions,
        grouped_overlap_count,
        load_dataset,
        mo,
        pd,
        run_baseline_validation,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 02 — Data preparation and baseline models

    This app retains exact-feature duplicates but keeps each feature signature within one partition. A button creates the grouped outer holdout and separate grouped validation split; another button triggers model fitting. **Opening the app does not split the full dataset, fit a model, or run SMOTE.**

    ## Why group exact input features?

    A conventional stratified row split can place identical feature rows on both sides. For the preserved conventional split, 227,845 training rows and 56,962 test rows contained 394 and 98 frauds, but 293 exact-feature signatures crossed the split. This benchmark instead hashes `X` only, then uses `StratifiedGroupKFold` to approximate class balance while assigning each feature group wholly to one partition. This is a learning comparison, not evidence of production or future-time generalization.
    """)
    return


@app.cell
def _(FEATURE_COLUMNS, load_dataset):
    dataset = load_dataset()
    features = dataset.loc[:, list(FEATURE_COLUMNS)]
    target = dataset["Class"]
    return features, target


@app.cell
def _(mo):
    partition_run_button = mo.ui.run_button(label="Create grouped partitions")
    partition_run_button
    return (partition_run_button,)


@app.cell
def _(create_grouped_partitions, features, mo, partition_run_button, target):
    mo.stop(
        not partition_run_button.value,
        mo.md("Click the button above to prepare the grouped training, validation, and reserved test partitions."),
    )
    partitions = create_grouped_partitions(features, target)
    return (partitions,)


@app.cell
def _(partitions, pd):
    partition_summary = pd.DataFrame(
        {
            "rows": [
                len(partitions.x_fit),
                len(partitions.x_validation),
                len(partitions.x_test),
            ],
            "frauds": [
                int(partitions.y_fit.sum()),
                int(partitions.y_validation.sum()),
                int(partitions.y_test.sum()),
            ],
            "fraud_rate_pct": [
                100 * partitions.y_fit.mean(),
                100 * partitions.y_validation.mean(),
                100 * partitions.y_test.mean(),
            ],
        },
        index=["model training", "validation", "reserved outer test"],
    )
    partition_summary
    return (partition_summary,)


@app.cell
def _(mo):
    outer_test_note = (
        "Outer-test class counts are shown only to verify stratification. "
        "The test partition is not used to fit preprocessing, select features, "
        "models, or thresholds, or to score models."
    )
    mo.md(outer_test_note)
    return


@app.cell
def _(grouped_overlap_count, mo, partitions):
    group_overlap_summary = {
        "training / validation": grouped_overlap_count(
            partitions.groups_fit,
            partitions.groups_validation,
        ),
        "training / outer test": grouped_overlap_count(
            partitions.groups_fit,
            partitions.groups_test,
        ),
        "validation / outer test": grouped_overlap_count(
            partitions.groups_validation,
            partitions.groups_test,
        ),
    }
    assert all(overlap == 0 for overlap in group_overlap_summary.values())
    mo.md(f"Exact-feature group overlap counts: `{group_overlap_summary}`")
    return (group_overlap_summary,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Leak-safe preprocessing

    The baseline pipeline order is **`StandardScaler -> SMOTE(random_state=42) -> classifier`**. Both scaling and resampling are fitted only on `model training`; validation and the reserved outer test retain natural prevalence. SMOTE is distance-based, so all 30 numeric inputs are standardized. Trees do not require scaling themselves, but use the same pipeline to keep this comparison's preprocessing boundary consistent.

    The classifier settings match the preserved benchmark: Decision Tree (`random_state=42`), Random Forest (`n_estimators=200`, `random_state=42`, `n_jobs=-1`), and AdaBoost (`random_state=42`).
    """)
    return


@app.cell
def _(mo):
    baseline_run_button = mo.ui.run_button(
        label="Train baselines and score validation"
    )
    baseline_run_button
    return (baseline_run_button,)


@app.cell
def _(baseline_run_button, mo, partitions, pd, run_baseline_validation):
    mo.stop(
        not baseline_run_button.value,
        mo.md("Click the button above to fit the three baselines and score validation."),
    )
    baseline_results, confusion_matrices = run_baseline_validation(
        partitions.x_fit,
        partitions.y_fit,
        partitions.x_validation,
        partitions.y_validation,
    )
    for model_name, matrix in confusion_matrices.items():
        print(f"{model_name} confusion matrix (rows=true, columns=predicted; labels 0, 1):")
        print(pd.DataFrame(matrix, index=[0, 1], columns=[0, 1]))
    baseline_results
    return baseline_results, confusion_matrices


@app.cell
def _(mo):
    mo.md(r"""
    ## Proposed MLP — not trained or validated

    The hypothesis is 30 inputs, hidden layers `(64, 32)`, ReLU, binary log-loss, Adam with `learning_rate_init=1e-3`, `alpha=1e-4`, batch size 256, `max_iter=100`, and seed 42. This is an unvalidated starting design, not a result. No MLP training or outer-test scoring is included in this migration. Do not enable early stopping inside a post-SMOTE estimator; validation belongs on the separate natural-prevalence partition.

    ## Interpretation boundary

    The preserved comparison has one validation split with 79 frauds. Random Forest is the strongest provisional balance on that split; AdaBoost has the highest recall but a severe false-alert burden. No cross-validation, threshold tuning, hyperparameter tuning, or final test result exists. See [the exact validation report](../reports/baseline_validation.md).
    """)
    return


if __name__ == "__main__":
    app.run()
