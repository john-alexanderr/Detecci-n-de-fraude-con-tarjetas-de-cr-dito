import marimo

__generated_with = "0.23.16"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from fraud_detection.data import load_dataset, raw_data_path
    from fraud_detection.splitting import create_grouped_partitions
    from fraud_detection.plots import (
        plot_amount_by_class,
        plot_amount_distribution,
        plot_class_prevalence,
        plot_correlation_heatmap,
        plot_iqr_flag_rates,
        plot_outlier_class_comparison,
        plot_pca_marginals,
        plot_relative_elapsed_time,
        plot_selected_pca_by_class,
        plot_target_associations,
        set_plot_theme,
    )

    set_plot_theme()
    return (
        load_dataset,
        mo,
        pd,
        create_grouped_partitions,
        plot_amount_by_class,
        plot_amount_distribution,
        plot_class_prevalence,
        plot_correlation_heatmap,
        plot_iqr_flag_rates,
        plot_outlier_class_comparison,
        plot_pca_marginals,
        plot_relative_elapsed_time,
        plot_selected_pca_by_class,
        plot_target_associations,
        raw_data_path,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 01 — Dataset understanding and exploratory analysis

    Build an evidence-led view of the public credit-card fraud dataset before modeling. This app downloads the pinned source only when the local CSV is absent and never edits or removes observations.

    ## Business framing and evaluation boundary

    A payment team may approve, decline, request additional verification, or review a transaction. Missing fraud may increase losses; flagging legitimate activity may create customer friction. The appropriate threshold depends on institution-specific costs and review capacity, which this dataset does not provide.

    The Kaggle source reports **284,807 transactions from two days in September 2013**, including **492 frauds**. `Class` is the binary target (`0` legitimate, `1` fraud). Accuracy alone is misleading under this severe imbalance; use Average Precision / PR-AUC, precision, recall, F1 or F2, and confusion matrices.

    The full-source cells below are limited to source inventory and the label prevalence needed for split stratification; they are not model-selection evidence. After an explicit button click, every model-relevant EDA plot and interpretation uses only a grouped development partition (outer training plus validation). The outer test is not passed to EDA functions or summarized in this app.

    ## Dataset scope

    There are 30 numeric inputs: `Time`, `Amount`, and `V1`–`V28`. `Time` is seconds elapsed since the first transaction—not a calendar timestamp. `V1`–`V28` are anonymized PCA components with no supplied direct business meaning. The dataset has no card/customer identifiers, merchant fields, demographic attributes, or behavioral sequence.
    """)
    return


@app.cell
def _(load_dataset, mo, raw_data_path):
    data_path = raw_data_path()
    df = load_dataset(data_path)
    mo.md(
        f"Using `{data_path.relative_to(data_path.parents[2])}`. "
        "The source file is reused if present; otherwise only `creditcard.csv` "
        "from Kaggle dataset version 3 is copied into `data/raw/`."
    )
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Full-source inventory and stratification diagnostics

    The following raw-source views describe schema, sample rows, summary statistics, missingness, duplicates, and label prevalence. They support input-integrity review and class stratification only; they are not evidence for selecting features, transformations, models, or thresholds.
    """)
    return


@app.cell
def _(df, pd):
    audit_overview = pd.DataFrame(
        {
            "Value": [
                len(df),
                df.shape[1],
                len(df.columns) - 1,
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
                int((df["Class"] == 1).sum()),
                float(df["Class"].mean() * 100),
            ]
        },
        index=[
            "Rows",
            "Columns",
            "Numeric inputs",
            "Missing cells",
            "Exact duplicate rows",
            "Fraud rows",
            "Fraud prevalence (%)",
        ],
    )
    audit_overview
    return (audit_overview,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.dtypes.astype(str).rename("dtype").to_frame()
    return


@app.cell
def _(df):
    df.describe().T
    return


@app.cell
def _(df):
    df["Class"].value_counts().rename("transactions").to_frame()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Source-inventory interpretation

    The preserved full-source profile reports 284,807 rows, 31 columns, 1,081 exact duplicate rows (0.3796%), and zero feature-conflicting-target groups. No missing values were detected. Candidate sentinel-like values are `Time = 999` (2 rows) and `Amount = 999` (7 rows); neither is confirmed missing. These are source-inventory diagnostics, not model-selection evidence. Keep duplicates and unusual values during this descriptive audit—these findings do not authorize deletion, recoding, or capping.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Development-only modeling EDA

    Creating the grouped partition is intentionally gated because it hashes the full feature table. The split cell exposes only the combined outer-development frame (training plus validation) to the EDA cells below. No outer-test counts or test-derived EDA results are displayed.
    """)
    return


@app.cell
def _(mo):
    development_eda_button = mo.ui.run_button(
        label="Create grouped development partition"
    )
    development_eda_button
    return (development_eda_button,)


@app.cell
def _(create_grouped_partitions, df, development_eda_button, mo, pd):
    mo.stop(
        not development_eda_button.value,
        mo.md("Click the button above to prepare development-only EDA."),
    )
    grouped_partitions = create_grouped_partitions(
        df.drop(columns=["Class"]),
        df["Class"],
    )
    development_df = pd.concat(
        [
            grouped_partitions.x_fit.assign(Class=grouped_partitions.y_fit),
            grouped_partitions.x_validation.assign(
                Class=grouped_partitions.y_validation
            ),
        ]
    ).sort_index()
    return (development_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ### EDA part 1 — Univariate distributions

    The plots use only the grouped development partition. Distribution differences alone do not establish predictive value.
    """)
    return


@app.cell
def _(development_df, plot_class_prevalence):
    plot_class_prevalence(development_df)
    return


@app.cell
def _(development_df, mo, plot_amount_distribution):
    mo.md(
        "The `log1p` axis compresses the development-partition display only; "
        "the source values remain unchanged."
    )
    plot_amount_distribution(development_df)
    return


@app.cell
def _(development_df, plot_pca_marginals):
    plot_pca_marginals(development_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### EDA part 2 — Bivariate comparisons

    Class-wise plots and target associations below use only the grouped development partition. They are exploratory evidence, not causal claims or automatic feature-selection rules.
    """)
    return


@app.cell
def _(development_df, plot_amount_by_class):
    plot_amount_by_class(development_df)
    return


@app.cell
def _(development_df, plot_selected_pca_by_class):
    plot_selected_pca_by_class(development_df)
    return


@app.cell
def _(development_df, mo, plot_target_associations):
    mo.md(
        "Pearson associations here are computed only on the development partition; "
        "they are not causal effects or proof of leakage."
    )
    plot_target_associations(development_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### EDA part 3 — Multivariate relationships

    The correlation view summarizes pairwise linear associations in the development partition only. Correlations alone do not establish feature utility or justify removal.
    """)
    return


@app.cell
def _(development_df, plot_correlation_heatmap):
    plot_correlation_heatmap(development_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### EDA part 4 — Relative elapsed-time pattern

    The development-partition bins below show counts and fraud rates over elapsed time. `Time` is not calendar time, so this cannot support clock-hour, weekday, holiday, or seasonality conclusions. Local variation also cannot establish temporal generalization.
    """)
    return


@app.cell
def _(development_df, plot_relative_elapsed_time):
    plot_relative_elapsed_time(development_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### EDA part 5 — IQR outlier inspection

    The 1.5 × IQR fences and class comparisons below are calculated only on the development partition. They flag values for review but do not distinguish valid rare transactions from data errors. No automatic row deletion or capping is applied.
    """)
    return


@app.cell
def _(development_df, plot_iqr_flag_rates):
    plot_iqr_flag_rates(development_df)
    return


@app.cell
def _(development_df, plot_outlier_class_comparison):
    plot_outlier_class_comparison(development_df)
    return


@app.cell
def _(development_df):
    classwise_outlier_quantiles = (
        development_df.groupby("Class")[["Amount", "V27"]].quantile(
            [0.5, 0.9, 0.99, 1.0]
        )
    )
    classwise_outlier_quantiles
    return (classwise_outlier_quantiles,)


if __name__ == "__main__":
    app.run()
