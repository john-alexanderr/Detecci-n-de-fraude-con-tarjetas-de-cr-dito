"""Gráficos descriptivos reutilizables para la aplicación Marimo de EDA."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_theme() -> None:
    """Aplica el estilo visual compartido por los gráficos de EDA."""
    sns.set_theme(style="whitegrid", context="notebook")


def plot_class_prevalence(frame: pd.DataFrame):
    fig, axis = plt.subplots(figsize=(6, 4))
    sns.countplot(data=frame, x="Class", ax=axis)
    axis.set_yscale("log")
    axis.set(
        title="Cantidad de transacciones por clase objetivo",
        xlabel="Class (0 = legítima, 1 = fraude)",
        ylabel="Cantidad (escala logarítmica)",
    )
    fig.tight_layout()
    return fig


def plot_class_share(frame: pd.DataFrame):
    counts = frame["Class"].value_counts().reindex([0, 1], fill_value=0)
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.pie(
        counts,
        colors=["#5b5bd6", "#e4572e"],
        autopct=lambda pct: (
            f"{pct:.1f}%".replace(".", ",")
            if pct >= 1
            else f"{pct:.3f}%".replace(".", ",")
        ),
        startangle=90,
        counterclock=False,
        pctdistance=1.06,
    )
    axis.legend(
        ["Legítima (clase 0)", "Fraude (clase 1)"],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    axis.set_title("Proporción de transacciones legítimas y fraudulentas")
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
        title="Distribución del importe de las transacciones en un eje log1p",
        xlabel="log1p(Amount)",
        ylabel="Densidad dentro de cada clase",
    )
    fig.tight_layout()
    return fig


def plot_amount_by_transaction(frame: pd.DataFrame):
    fig, axis = plt.subplots(figsize=(11, 5))
    points = axis.scatter(
        frame["Amount"],
        frame.index,
        c=frame["Class"],
        cmap="plasma",
        s=8,
        alpha=0.7,
        linewidths=0,
        rasterized=True,
    )
    colorbar = fig.colorbar(points, ax=axis)
    colorbar.set_label("Class (0 = legítima, 1 = fraude)")
    axis.set(
        title="Distribución del importe de las transacciones",
        xlabel="Importe de la transacción (Amount)",
        ylabel="Índice de transacción",
    )
    fig.tight_layout()
    return fig


def plot_fraud_amount_by_transaction(frame: pd.DataFrame):
    fraud = frame.loc[frame["Class"] == 1]
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.scatter(
        fraud["Amount"],
        fraud.index,
        color="#5b5bd6",
        s=14,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )
    axis.set(
        title="Distribución del importe de las transacciones fraudulentas",
        xlabel="Importe de la transacción (Amount)",
        ylabel="Índice de transacción",
    )
    fig.tight_layout()
    return fig


def plot_confusion_matrix_concept():
    """Dibuja una matriz de confusión genérica con las cuatro combinaciones."""
    fig, axis = plt.subplots(figsize=(7, 5.5))
    cells = [
        ["TN", "FP"],
        ["FN", "TP"],
    ]
    descriptions = [
        ["Legítima correcta", "Falsa alarma"],
        ["Fraude no detectado", "Fraude detectado"],
    ]
    colors = [
        ["#9bc57a", "#bcd9e8"],
        ["#bcd9e8", "#9bc57a"],
    ]
    for row in range(2):
        for column in range(2):
            axis.add_patch(
                plt.Rectangle(
                    (column, 1 - row),
                    1,
                    1,
                    facecolor=colors[row][column],
                    edgecolor="white",
                    linewidth=3,
                )
            )
            axis.text(
                column + 0.5,
                1 - row + 0.62,
                cells[row][column],
                ha="center",
                va="center",
                fontsize=24,
                fontweight="bold",
                color="#1f2d3d",
            )
            axis.text(
                column + 0.5,
                1 - row + 0.32,
                descriptions[row][column],
                ha="center",
                va="center",
                fontsize=10,
                color="#1f2d3d",
            )
    axis.set_xlim(0, 2)
    axis.set_ylim(0, 2)
    axis.set_xticks([0.5, 1.5], ["Predicho: legítima (0)", "Predicho: fraude (1)"])
    axis.set_yticks([1.5, 0.5], ["Real: legítima (0)", "Real: fraude (1)"])
    axis.tick_params(length=0)
    axis.set_xlabel("Clase predicha", fontsize=12, fontweight="bold")
    axis.set_ylabel("Clase real", fontsize=12, fontweight="bold")
    axis.set_title("Matriz de confusión: cómo se lee", fontsize=14)
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(confusion_matrices: dict[str, list[list[int]]]):
    """Dibuja una matriz de confusión por modelo con las siglas y los recuentos."""
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    fig, axes = plt.subplots(
        1, len(confusion_matrices), figsize=(6 * len(confusion_matrices), 5)
    )
    axes = np.atleast_1d(axes)
    for axis, (model_name, matrix) in zip(
        axes, confusion_matrices.items(), strict=True
    ):
        counts = np.asarray(matrix)
        annotations = np.empty(counts.shape, dtype=object)
        for row in range(counts.shape[0]):
            for column in range(counts.shape[1]):
                value = f"{counts[row, column]:,}".replace(",", ".")
                annotations[row, column] = f"{labels[row, column]}\n{value}"
        sns.heatmap(
            counts,
            annot=annotations,
            fmt="",
            cmap="Blues",
            cbar=False,
            square=True,
            linewidths=2,
            linecolor="white",
            ax=axis,
            annot_kws={"fontsize": 13, "fontweight": "bold"},
        )
        axis.set_xticklabels(["Predicho: legítima", "Predicho: fraude"], rotation=0)
        axis.set_yticklabels(["Real: legítima", "Real: fraude"], rotation=0)
        axis.set_title(model_name)
        axis.set_xlabel("Clase predicha", fontsize=11)
        axis.set_ylabel("Clase real", fontsize=11)
    fig.tight_layout()
    return fig


def plot_pca_marginals(frame: pd.DataFrame):
    components = [f"V{index}" for index in range(1, 29)]
    figure, axes = plt.subplots(4, 7, figsize=(18, 12))
    for component, axis in zip(components, axes.flat, strict=True):
        sns.histplot(data=frame, x=component, bins=50, stat="density", ax=axis)
        axis.set_title(component)
    figure.suptitle("Distribuciones marginales de los componentes PCA anonimizados", y=1.01)
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
        title="log1p(Amount) por clase objetivo",
        xlabel="Class (0 = legítima, 1 = fraude)",
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
        axis.set_title(f"{component} por clase objetivo")
    figure.tight_layout()
    return figure


def plot_target_associations(frame: pd.DataFrame):
    associations = frame.corr(numeric_only=True)["Class"].drop("Class").sort_values()
    fig, axis = plt.subplots(figsize=(9, 10))
    associations.plot.barh(ax=axis, color="steelblue")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(
        title="Asociación de Pearson de las entradas numéricas con Class",
        xlabel="Correlación de Pearson (asociación, no causalidad)",
        ylabel="Variable de entrada",
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
    axis.set_title("Correlación de Pearson entre las entradas y el objetivo")
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
        title="Cantidad de transacciones y fraudes por intervalo de hora transcurrida",
        xlabel="",
        ylabel="Cantidad de transacciones y fraudes",
    )
    hourly_summary["fraud_rate"].plot(kind="line", marker="o", ax=axes[1])
    axes[1].set(
        title="Tasa de fraude por intervalo de hora transcurrida",
        xlabel="Hora transcurrida relativa (no hora de calendario)",
        ylabel="Tasa de fraude",
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
        title="Filas fuera de los límites de 1.5 × IQR por variable numérica",
        xlabel="Filas marcadas (%)",
        ylabel="Variable de entrada",
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
    axes[0].set(
        title="log1p(Amount) por clase", xlabel="Class", ylabel="log1p(Amount)"
    )
    sns.boxplot(
        data=comparison,
        x="Class",
        y="V27",
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set(title="V27 por clase", xlabel="Class", ylabel="V27")
    figure.tight_layout()
    return figure
