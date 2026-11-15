"""Genera la matriz de confusión del MLP a partir de sus recuentos verificados."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TRUE_NEGATIVES = 56_830
FALSE_POSITIVES = 33
FALSE_NEGATIVES = 16
TRUE_POSITIVES = 83

WIDTH_PX = 1_500
HEIGHT_PX = 840
DPI = 150
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "site"
    / "charts"
    / "matriz-confusion.png"
)

BACKGROUND = "#0B1020"
PANEL = "#0E1530"
TEXT = "#E6EDF7"
LABEL = "#C9D6F0"
MUTED = "#6B7AA6"
GREEN = "#34D399"
AMBER = "#FBBF24"
ROSE = "#FB7185"
CELL_TN = "#123B3A"
CELL_FP = "#3D351F"
CELL_FN = "#432637"
CELL_TP = "#12443D"


def format_count(value: int) -> str:
    """Devuelve un recuento con separador de miles usado en el proyecto."""
    return f"{value:,}".replace(",", ".")


def format_percentage(numerator: int, denominator: int) -> str:
    """Calcula y formatea el porcentaje de una celda dentro de su fila real."""
    return f"{100 * numerator / denominator:.2f}".replace(".", ",")


def add_figure_text(
    figure: plt.Figure,
    x_px: float,
    y_px: float,
    text: str,
    **kwargs: object,
) -> None:
    """Añade texto usando coordenadas en píxeles desde la esquina superior."""
    kwargs.setdefault("color", TEXT)
    figure.text(
        x_px / WIDTH_PX,
        1 - y_px / HEIGHT_PX,
        text,
        fontfamily="DejaVu Sans",
        **kwargs,
    )


def generate_chart() -> None:
    """Dibuja el PNG sin descargar datos ni ajustar modelos."""
    legitimate_total = TRUE_NEGATIVES + FALSE_POSITIVES
    fraud_total = FALSE_NEGATIVES + TRUE_POSITIVES
    total_rows = legitimate_total + fraud_total

    if total_rows != 56_962 or fraud_total != 99:
        raise ValueError("Los recuentos no coinciden con el test reservado verificado.")

    figure = plt.figure(
        figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI),
        dpi=DPI,
        facecolor=BACKGROUND,
    )

    add_figure_text(
        figure,
        68,
        44,
        "MLP (64,32) — test reservado",
        fontsize=20,
        fontweight="bold",
        color=LABEL,
        va="center",
    )

    matrix_left = 450
    matrix_top = 166
    matrix_width = 630
    matrix_height = 563
    matrix_axes = figure.add_axes(
        [
            matrix_left / WIDTH_PX,
            1 - (matrix_top + matrix_height) / HEIGHT_PX,
            matrix_width / WIDTH_PX,
            matrix_height / HEIGHT_PX,
        ],
        facecolor=PANEL,
    )
    matrix_axes.set_xlim(0, 2)
    matrix_axes.set_ylim(0, 2)
    matrix_axes.axis("off")

    cells = (
        (0, 1, CELL_TN, "#1C6B5C", "TN · Verdadero negativo", TRUE_NEGATIVES),
        (1, 1, CELL_FP, "#7B5A12", "FP · Falsa alarma", FALSE_POSITIVES),
        (0, 0, CELL_FN, "#8B3D52", "FN · Falso negativo", FALSE_NEGATIVES),
        (1, 0, CELL_TP, "#1C6B5C", "TP · Verdadero positivo", TRUE_POSITIVES),
    )
    row_totals = (legitimate_total, fraud_total)
    row_counts = (
        (TRUE_NEGATIVES, FALSE_POSITIVES),
        (FALSE_NEGATIVES, TRUE_POSITIVES),
    )
    stripe_height = 0.057

    for column, row, fill, border, title, count in cells:
        matrix_axes.add_patch(
            Rectangle(
                (column, row),
                1,
                1,
                facecolor=fill,
                edgecolor=border,
                linewidth=1.2,
            )
        )
        stripe_color = GREEN if title.startswith(("TN", "TP")) else (
            AMBER if title.startswith("FP") else ROSE
        )
        matrix_axes.add_patch(
            Rectangle(
                (column, row + 1 - stripe_height),
                1,
                stripe_height,
                facecolor=stripe_color,
                edgecolor="none",
            )
        )

        row_index = 0 if row == 1 else 1
        denominator = row_totals[row_index]
        row_values = row_counts[row_index]
        column_index = column
        percentage = format_percentage(row_values[column_index], denominator)

        matrix_axes.text(
            column + 0.5,
            row + 0.86,
            title,
            color=stripe_color,
            fontsize=10.6,
            fontweight="bold",
            ha="center",
            va="center",
        )
        matrix_axes.text(
            column + 0.5,
            row + 0.60,
            format_count(count),
            color=TEXT,
            fontsize=28,
            fontweight="bold",
            ha="center",
            va="center",
        )
        matrix_axes.text(
            column + 0.5,
            row + 0.26,
            f"{percentage} % del bloque real",
            color=TEXT,
            fontsize=10.2,
            ha="center",
            va="center",
        )

    add_figure_text(
        figure,
        607,
        128,
        "Predice 0",
        fontsize=13.5,
        fontweight="bold",
        color=LABEL,
        ha="center",
        va="center",
    )
    add_figure_text(
        figure,
        923,
        128,
        "Predice 1",
        fontsize=13.5,
        fontweight="bold",
        color=LABEL,
        ha="center",
        va="center",
    )
    add_figure_text(
        figure,
        164,
        306,
        "Real 0 · legítima",
        fontsize=12.8,
        fontweight="bold",
        color=LABEL,
        ha="left",
        va="center",
    )
    add_figure_text(
        figure,
        185,
        585,
        "Real 1 · fraude",
        fontsize=12.8,
        fontweight="bold",
        color=LABEL,
        ha="left",
        va="center",
    )

    side_labels = (
        (276, f"{format_count(TRUE_POSITIVES)} / {format_count(fraud_total)}", GREEN, "fraudes detectados"),
        (432, f"{format_count(FALSE_NEGATIVES)} / {format_count(fraud_total)}", ROSE, "fraudes no detectados"),
        (
            588,
            f"{format_count(FALSE_POSITIVES)} / {format_count(legitimate_total)}",
            AMBER,
            "falsas alarmas",
        ),
    )
    for y_px, value, color, description in side_labels:
        add_figure_text(
            figure,
            1162,
            y_px,
            value,
            fontsize=20,
            fontweight="bold",
            color=color,
            va="center",
        )
        add_figure_text(
            figure,
            1162,
            y_px + 41,
            description,
            fontsize=11.4,
            color=MUTED,
            va="center",
        )

    add_figure_text(
        figure,
        68,
        789,
        "Test reservado: "
        f"{format_count(total_rows)} filas · {format_count(fraud_total)} fraudes · "
        "el % corresponde al bloque real de cada fila",
        fontsize=11.5,
        color=MUTED,
        va="center",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        dpi=DPI,
        facecolor=BACKGROUND,
        edgecolor=BACKGROUND,
        transparent=False,
    )
    plt.close(figure)
    print(
        f"Gráfico generado: {OUTPUT_PATH} ({WIDTH_PX}x{HEIGHT_PX} px); "
        f"filas={format_count(total_rows)}, legítimas={format_count(legitimate_total)}, "
        f"fraudes={format_count(fraud_total)}."
    )


if __name__ == "__main__":
    generate_chart()
