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
    # 03 — Validación cruzada agrupada del MLP y evaluación final

    La validación cruzada documentada del MLP, el entrenamiento y reajuste con los datos de desarrollo y una evaluación final en la prueba reservada ya están completos. Abrir esta aplicación no carga el CSV, no inicia una búsqueda, no ajusta un modelo ni evalúa la prueba reservada. El botón de abajo repetiría la búsqueda y volvería a evaluar la prueba reservada; no usarlo para seguir iterando. Revisar los resultados registrados y el límite de la evaluación que aparecen abajo.

    ## ¿Por qué un MLP, escalado y SMOTE?

    Las 30 variables de entrada numéricas tienen escalas muy distintas, y un MLP es sensible a la escala de las variables. Por eso el pipeline aplica **`StandardScaler -> SMOTE(random_state=42) -> MLPClassifier`**. El pipeline de imbalanced-learn está dentro de la validación cruzada, así que el escalado y el remuestreo se ajustan solo con los datos de entrenamiento de cada fold; cada fold de validación mantiene la prevalencia natural. La parada temprana está desactivada porque su división de validación interna se tomaría de los datos de entrenamiento posteriores a SMOTE.

    ## Búsqueda agrupada acotada

    Las particiones internas de entrenamiento del modelo y validación se combinan como datos de desarrollo; la prueba externa reservada no se incluye. La búsqueda con `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=44)` recibe los identificadores de grupo de desarrollo, lo que mantiene juntos los duplicados exactos de variables. Tres folds ofrecen una estimación inicial práctica con 393 fraudes de desarrollo; no hacen que el resultado sea exhaustivo.

    Los dos candidatos difieren en el tamaño de la red y la regularización L2:

    | Candidato | Capas ocultas | `alpha` | `learning_rate_init` |
    | --- | --- | ---: | ---: |
    | 1 | `(32,)` | `1e-3` | `1e-3` |
    | 2 | `(64, 32)` | `1e-4` | `1e-3` |

    Ambos usan ReLU, Adam, tamaño de lote 256, `max_iter=100`, `early_stopping=False` y semilla 42. La selección de la búsqueda usa el Average Precision medio de los folds. Esta es una **búsqueda inicial acotada**, no un óptimo exhaustivo.

    ## Límite de la prueba final

    Después de la validación cruzada, `GridSearchCV(refit=True)` ajusta el pipeline seleccionado con todas las filas de desarrollo. Luego, la prueba externa se evalúa una sola vez y nunca se usa para seleccionar parámetros ni ajustar un umbral. Average Precision usa `predict_proba`; la precisión, el recall (exhaustividad), la métrica F2 y los recuentos de la matriz de confusión usan las etiquetas por defecto de `predict` del estimador. La prueba contiene 56,962 filas y solo 99 fraudes, así que sus métricas son una única estimación final sobre datos reservados, no una prueba del rendimiento en producción.
    """)
    return


@app.cell
def _(mo):
    mlp_run_button = mo.ui.run_button(
        label="Ejecutar la búsqueda acotada del MLP y la prueba final"
    )
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
            "Hacer clic en el botón de arriba para ejecutar la validación cruzada "
            "agrupada y luego evaluar la prueba reservada una sola vez."
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
    ## Resultados de la búsqueda
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
    ## Una evaluación final en la prueba reservada
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
    ## Guía de interpretación

    La búsqueda completada con tres folds seleccionó el candidato 2, `(64, 32)` con `alpha=0.0001` y `learning_rate_init=0.001`, con un Average Precision medio de validación cruzada de `0.7892942255438676` y una desviación estándar de `0.024821041864665654`. El candidato 1, `(32,)` con `alpha=0.001`, obtuvo una media de `0.7889909900343617` y una desviación estándar de `0.016266031474164548`. El candidato 2 gana la búsqueda, pero su ventaja media es mínima frente a la variación entre folds y no demuestra una superioridad decisiva de la arquitectura.

    Después de reajustar el pipeline seleccionado con todos los datos de desarrollo, la prueba reservada se evaluó una sola vez: Average Precision `0.8203794692697799`, precisión `0.7155172413793104`, recall (exhaustividad) `0.8383838383838383` y F2 `0.810546875` sobre 56,962 filas con 99 fraudes. No hubo ajuste de umbral; `predict` usa las etiquetas de decisión por defecto, no un punto de operación aprobado por el negocio. Esta es una estimación incierta sobre datos reservados, no evidencia del rendimiento con fraude actual ni en producción. No volver a usar la prueba externa para iterar. Los modelos base de árboles tienen resultados solo de validación y no son directamente comparables porque usan una partición distinta. Consultar el [informe completo de evaluación del MLP](../reports/mlp_final_evaluation.md) para ver todos los resultados y la matriz de confusión.
    """)
    return


if __name__ == "__main__":
    app.run()
