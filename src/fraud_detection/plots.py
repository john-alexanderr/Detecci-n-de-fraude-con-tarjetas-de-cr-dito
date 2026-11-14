"""Reusable descriptive plots for the EDA Marimo application."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_theme() -> None:
    """Apply the shared visual style used by the EDA charts."""
    sns.set_theme(style="whitegrid", context="notebook")


def plot_class_prevalence(frame: pd.DataFrame):
    fig, axis = plt.subplots(figsize=(6, 4))
    sns.countplot(data=frame, x="Class", ax=axis)
    axis.set_yscale("log")
    axis.set(
        title="Transaction counts by target class",
        xlabel="Class (0 = legitimate, 1 = fraud)",
        ylabel="Count (log scale)",
    )
    fig.tight_layout()
    return fig


def plot_amount_distribution(frame: pd.DataFrame):
    amount_plot = frame[["Amount", "Class"]].copy()
    amount_plot["log1p_amount"] = np.log1p(amount_plot["Amount"])
    fig, axis = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=amount_plot,
        x="log1p_amount",
        hue="Class",
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        ax=axis,
    )
    axis.set(
        title="Transaction amount distribution on a log1p axis",
        xlabel="log1p(Amount)",
        ylabel="Within-class density",
    )
    fig.tight_layout()
    return fig


def plot_pca_marginals(frame: pd.DataFrame):
    components = [f"V{index}" for index in range(1, 29)]
    figure, axes = plt.subplots(4, 7, figsize=(18, 12))
    for component, axis in zip(components, axes.flat, strict=True):
        sns.histplot(data=frame, x=component, bins=50, stat="density", ax=axis)
        axis.set_title(component)
    figure.suptitle("Marginal distributions of anonymized PCA components", y=1.01)
    figure.tight_layout()
    return figure


def plot_amount_by_class(frame: pd.DataFrame):
    amount_plot = frame[["Amount", "Class"]].copy()
    amount_plot["log1p_amount"] = np.log1p(amount_plot["Amount"])
    fig, axis = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=amount_plot,
        x="Class",
        y="log1p_amount",
        showfliers=True,
        ax=axis,
    )
    axis.set(
        title="log1p(Amount) by target class",
        xlabel="Class (0 = legitimate, 1 = fraud)",
        ylabel="log1p(Amount)",
    )
    fig.tight_layout()
    return fig


def plot_selected_pca_by_class(frame: pd.DataFrame):
    components = ["V14", "V17", "V12", "V10"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for component, axis in zip(components, axes.flat, strict=True):
        sns.boxplot(
            data=frame,
            x="Class",
            y=component,
            showfliers=False,
            ax=axis,
        )
        axis.set_title(f"{component} by target class")
    figure.tight_layout()
    return figure


def plot_target_associations(frame: pd.DataFrame):
    associations = frame.corr(numeric_only=True)["Class"].drop("Class").sort_values()
    fig, axis = plt.subplots(figsize=(9, 10))
    associations.plot.barh(ax=axis, color="steelblue")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(
        title="Pearson association of numeric inputs with Class",
        xlabel="Pearson correlation (association, not causation)",
        ylabel="Input feature",
    )
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(frame: pd.DataFrame):
    correlation = frame.corr(numeric_only=True)
    fig, axis = plt.subplots(figsize=(15, 12))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        square=True,
        xticklabels=True,
        yticklabels=True,
        ax=axis,
    )
    axis.set_title("Pearson correlation across inputs and target")
    fig.tight_layout()
    return fig


def plot_relative_elapsed_time(frame: pd.DataFrame):
    relative_hour = np.floor(frame["Time"] / 3600).astype(int)
    hourly_summary = frame.assign(relative_hour=relative_hour).groupby("relative_hour")[
        "Class"
    ].agg(transaction_count="count", fraud_count="sum", fraud_rate="mean")

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    hourly_summary[["transaction_count", "fraud_count"]].plot(
        kind="bar",
        ax=axes[0],
        width=0.85,
    )
    axes[0].set(
        title="Counts by relative elapsed-hour bin",
        xlabel="",
        ylabel="Transaction and fraud counts",
    )
    hourly_summary["fraud_rate"].plot(kind="line", marker="o", ax=axes[1])
    axes[1].set(
        title="Fraud rate by relative elapsed-hour bin",
        xlabel="Relative elapsed hour (not calendar hour)",
        ylabel="Fraud rate",
    )
    figure.tight_layout()
    return figure


def plot_iqr_flag_rates(frame: pd.DataFrame):
    feature_frame = frame.drop(columns=["Class"])
    lower_quartile = feature_frame.quantile(0.25)
    upper_quartile = feature_frame.quantile(0.75)
    interquartile_range = upper_quartile - lower_quartile
    outside_fences = (
        (feature_frame.lt(lower_quartile - 1.5 * interquartile_range))
        | (feature_frame.gt(upper_quartile + 1.5 * interquartile_range))
    )
    rates = outside_fences.mean().mul(100).sort_values()

    fig, axis = plt.subplots(figsize=(9, 10))
    rates.plot.barh(ax=axis, color="darkorange")
    axis.set(
        title="Rows outside 1.5 × IQR fences by numeric input",
        xlabel="Rows flagged (%)",
        ylabel="Input feature",
    )
    fig.tight_layout()
    return fig


def plot_outlier_class_comparison(frame: pd.DataFrame):
    comparison = frame[["Amount", "V27", "Class"]].copy()
    comparison["log1p_amount"] = np.log1p(comparison["Amount"])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(
        data=comparison,
        x="Class",
        y="log1p_amount",
        showfliers=False,
        ax=axes[0],
    )
    axes[0].set(title="log1p(Amount) by class", xlabel="Class", ylabel="log1p(Amount)")
    sns.boxplot(
        data=comparison,
        x="Class",
        y="V27",
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set(title="V27 by class", xlabel="Class", ylabel="V27")
    figure.tight_layout()
    return figure
