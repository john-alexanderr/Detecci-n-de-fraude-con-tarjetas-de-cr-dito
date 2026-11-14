"""Búsqueda acotada con validación cruzada agrupada y evaluación final en la partición de prueba reservada para un MLP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from fraud_detection.splitting import GroupedPartitions

MLP_PARAMETER_GRID = (
    {
        "classifier__hidden_layer_sizes": [(32,)],
        "classifier__alpha": [1e-3],
        "classifier__learning_rate_init": [1e-3],
    },
    {
        "classifier__hidden_layer_sizes": [(64, 32)],
        "classifier__alpha": [1e-4],
        "classifier__learning_rate_init": [1e-3],
    },
)


@dataclass(frozen=True)
class CandidateCVSummary:
    """Métricas de validación cruzada para un candidato MLP configurado."""

    params: dict[str, object]
    mean_average_precision: float
    std_average_precision: float
    fold_count: int


@dataclass(frozen=True)
class MLPSearchResult:
    """Estimador seleccionado y resumen de validación cruzada de la búsqueda acotada."""

    best_params: dict[str, object]
    best_average_precision: float
    candidates: tuple[CandidateCVSummary, ...]
    best_estimator: Pipeline


@dataclass(frozen=True)
class MLPTestEvaluation:
    """Una evaluación final sobre la partición de prueba externa reservada."""

    row_count: int
    fraud_count: int
    average_precision: float
    precision: float
    recall: float
    f2: float
    tn: int
    fp: int
    fn: int
    tp: int
    preview: pd.DataFrame


def build_mlp_pipeline() -> Pipeline:
    """Construye un pipeline local a cada fold con escalador, muestreador y MLP."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            (
                "classifier",
                MLPClassifier(
                    activation="relu",
                    solver="adam",
                    batch_size=256,
                    max_iter=100,
                    early_stopping=False,
                    random_state=42,
                ),
            ),
        ]
    )


def build_mlp_search() -> GridSearchCV:
    """Crea la búsqueda agrupada documentada de dos candidatos y tres folds."""
    parameter_grid = [
        {key: list(values) for key, values in candidate.items()}
        for candidate in MLP_PARAMETER_GRID
    ]
    return GridSearchCV(
        estimator=build_mlp_pipeline(),
        param_grid=parameter_grid,
        scoring="average_precision",
        cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=44),
        refit=True,
        n_jobs=1,
        pre_dispatch=1,
        error_score="raise",
    )


def combine_development_partitions(
    partitions: GroupedPartitions,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Combina solo las filas de entrenamiento y validación internas para la validación cruzada agrupada."""
    features = pd.concat(
        [partitions.x_fit, partitions.x_validation],
        axis=0,
    )
    target = pd.concat(
        [partitions.y_fit, partitions.y_validation],
        axis=0,
    )
    groups = np.concatenate(
        [partitions.groups_fit, partitions.groups_validation]
    )
    if not (len(features) == len(target) == len(groups)):
        raise ValueError(
            "Las variables de entrada, el objetivo y los grupos de desarrollo deben estar alineados."
        )
    return features, target, groups


def _unprefixed_params(params: dict[str, object]) -> dict[str, object]:
    prefix = "classifier__"
    return {
        key.removeprefix(prefix): value
        for key, value in params.items()
    }


def run_mlp_search(partitions: GroupedPartitions) -> MLPSearchResult:
    """Ajusta los hiperparámetros con los datos de desarrollo combinados y luego reajusta el pipeline seleccionado."""
    x_development, y_development, groups_development = (
        combine_development_partitions(partitions)
    )
    search = build_mlp_search()
    search.fit(x_development, y_development, groups=groups_development)

    fold_count = sum(
        key.startswith("split") and key.endswith("_test_score")
        for key in search.cv_results_
    )
    candidate_summaries = tuple(
        CandidateCVSummary(
            params=_unprefixed_params(params),
            mean_average_precision=float(mean_score),
            std_average_precision=float(std_score),
            fold_count=fold_count,
        )
        for params, mean_score, std_score in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
            strict=True,
        )
    )
    return MLPSearchResult(
        best_params=_unprefixed_params(search.best_params_),
        best_average_precision=float(search.best_score_),
        candidates=candidate_summaries,
        best_estimator=search.best_estimator_,
    )


def evaluate_mlp_test(
    best_estimator: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    preview_rows: int = 10,
) -> MLPTestEvaluation:
    """Evalúa la prueba reservada una sola vez usando probabilidades y etiquetas por defecto."""
    if len(x_test) != len(y_test):
        raise ValueError(
            "Las variables de entrada y el objetivo de prueba deben contener las mismas filas."
        )
    if len(y_test) == 0:
        raise ValueError("La partición de prueba reservada no debe estar vacía.")
    if preview_rows < 0:
        raise ValueError("preview_rows debe ser no negativo.")

    actual = np.asarray(y_test)
    probabilities = best_estimator.predict_proba(x_test)[:, 1]
    predicted = best_estimator.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()
    preview = pd.DataFrame(
        {
            "actual_label": actual[:preview_rows],
            "fraud_probability": probabilities[:preview_rows],
            "predicted_label": predicted[:preview_rows],
        }
    )
    return MLPTestEvaluation(
        row_count=int(len(actual)),
        fraud_count=int(np.count_nonzero(actual == 1)),
        average_precision=float(average_precision_score(actual, probabilities)),
        precision=float(precision_score(actual, predicted, zero_division=0)),
        recall=float(recall_score(actual, predicted, zero_division=0)),
        f2=float(fbeta_score(actual, predicted, beta=2, zero_division=0)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        preview=preview,
    )
