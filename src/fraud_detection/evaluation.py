"""Utilidades de métricas solo para entrenamiento y validación."""

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)

from fraud_detection.models import build_baseline_pipelines


def summarize_predictions(
    y_true: pd.Series,
    y_score,
    y_pred,
) -> dict[str, float | list[list[int]]]:
    """Resume el ranking de probabilidades y las predicciones con el umbral por defecto."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "average_precision": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }


def run_baseline_validation(
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[pd.DataFrame, dict[str, list[list[int]]]]:
    """Ajusta los modelos base con datos de entrenamiento y los evalúa en validación con prevalencia natural."""
    result_rows = []
    confusion_matrices: dict[str, list[list[int]]] = {}

    for model_name, pipeline in build_baseline_pipelines().items():
        pipeline.fit(x_fit, y_fit)
        validation_scores = pipeline.predict_proba(x_validation)[:, 1]
        validation_predictions = pipeline.predict(x_validation)
        metrics = summarize_predictions(
            y_validation,
            validation_scores,
            validation_predictions,
        )
        confusion_matrices[model_name] = metrics.pop("confusion_matrix")
        result_rows.append({"model": model_name, **metrics})

    results = pd.DataFrame(result_rows).set_index("model")
    return results, confusion_matrices
