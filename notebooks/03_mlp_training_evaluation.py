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
    # 03 — Entrenamiento del MLP: validación cruzada y prueba final

    Este notebook muestra cómo se entrena y elige el MLP, paso a paso. Todo el proceso ya está completado y registrado: abrir la aplicación no carga el CSV, no entrena ni evalúa nada. El botón repite la búsqueda y volvería a evaluar la prueba reservada: **no usarlo para seguir iterando**.

    ## Paso 1 — Separar el desarrollo de la prueba reservada

    La búsqueda usa solo el desarrollo agrupado —entrenamiento + validación, 227.845 filas con 393 fraudes—. La prueba externa (56.962 filas, 99 fraudes) queda fuera y se evalúa una sola vez al final.

    ## Paso 2 — Validar con folds agrupados

    `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=44)` reparte el desarrollo en tres folds. Los duplicados exactos de variables quedan siempre juntos y cada fold conserva una proporción similar de fraudes. Tres folds con 393 fraudes dan una estimación inicial práctica, no un resultado exhaustivo.

    ## Paso 3 — Comparar dos candidatos

    | Candidato | Capas ocultas | `alpha` | `learning_rate_init` |
    | --- | --- | ---: | ---: |
    | 1 | `(32,)` | `1e-3` | `1e-3` |
    | 2 | `(64, 32)` | `1e-4` | `1e-3` |

    Ambos usan ReLU, Adam, tamaño de lote 256, `max_iter=100`, `early_stopping=False` y semilla 42. El pipeline es el mismo en los dos: `StandardScaler → SMOTE → MLPClassifier`, ajustado dentro de cada fold. La parada temprana está desactivada porque su validación interna se tomaría de los datos ya remuestreados por SMOTE.

    Así se construye el pipeline con el candidato ganador:

    ```python
    Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-4,
            learning_rate_init=1e-3,
            activation="relu",
            solver="adam",
            batch_size=256,
            max_iter=100,
            early_stopping=False,
            random_state=42,
        )),
    ])
    ```

    ## Paso 4 — Elegir por Average Precision

    La selección usa el Average Precision medio de los tres folds. Esta métrica resume la curva precisión-recall y castiga que los fraudes queden abajo en el ranking de probabilidad. Es una **búsqueda inicial acotada**, no un óptimo exhaustivo.

    ## Paso 5 — Reajustar y evaluar la prueba una sola vez

    `GridSearchCV(refit=True)` reajusta el pipeline ganador con todas las filas de desarrollo. Luego, la prueba externa se evalúa una única vez y nunca se usa para seleccionar parámetros ni ajustar un umbral. Average Precision usa `predict_proba`; la precisión, el recall, la métrica F2 y la matriz de confusión usan las etiquetas por defecto de `predict`. La prueba contiene 56.962 filas y solo 99 fraudes, así que sus métricas son una única estimación final sobre datos reservados, no una prueba del rendimiento en producción.
    """)
    return


@app.cell
def _(mo):
    mo.mermaid("""
    flowchart TD
        A["Desarrollo: 227.845 filas · 393 fraudes"] --> B["StratifiedGroupKFold · 3 folds"]
        B --> C["Candidato 1: capas (32,)"]
        B --> D["Candidato 2: capas (64, 32)"]
        C --> E["Average Precision media de los folds"]
        D --> E
        E --> F["Reajuste con todo el desarrollo"]
        F --> G["Prueba reservada: una sola evaluación"]
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
    ## Resultados de la validación cruzada

    La tabla compara los dos candidatos: la columna `selected` marca el elegido y `mean_cv_average_precision` promedia el Average Precision de los tres folds.
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
    ## Resultados en la prueba reservada
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

    Cómo leer las métricas:

    - **Average Precision (AP)**: área bajo la curva precisión-recall; mide qué tan bien ordena el modelo los fraudes por encima de las legítimas.
    - **Precisión**: de las alertas que marcó el modelo, qué fracción era fraude real.
    - **Recall (exhaustividad)**: de todos los fraudes reales, qué fracción fue detectada.
    - **F2**: combina precisión y recall dando el doble de peso al recall.
    - **Matriz de confusión**: TN y TP son aciertos, FP son falsas alarmas y FN son fraudes no detectados.

    La búsqueda completada con tres folds seleccionó el candidato 2, `(64, 32)` con `alpha=0.0001` y `learning_rate_init=0.001`, con un Average Precision medio de validación cruzada de `0.7892942255438676` y una desviación estándar de `0.024821041864665654`. El candidato 1, `(32,)` con `alpha=0.001`, obtuvo una media de `0.7889909900343617` y una desviación estándar de `0.016266031474164548`. El candidato 2 gana la búsqueda, pero su ventaja media es mínima frente a la variación entre folds y no demuestra una superioridad decisiva de la arquitectura.

    Después de reajustar el pipeline seleccionado con todos los datos de desarrollo, la prueba reservada se evaluó una sola vez: Average Precision `0.8203794692697799`, precisión `0.7155172413793104`, recall (exhaustividad) `0.8383838383838383` y F2 `0.810546875` sobre 56,962 filas con 99 fraudes. No hubo ajuste de umbral; `predict` usa las etiquetas de decisión por defecto, no un punto de operación aprobado por el negocio. Esta es una estimación incierta sobre datos reservados, no evidencia del rendimiento con fraude actual ni en producción. No volver a usar la prueba externa para iterar. Los modelos base de árboles tienen resultados solo de validación y no son directamente comparables porque usan una partición distinta. Consultar el [informe completo de evaluación del MLP](../reports/mlp_final_evaluation.md) para ver todos los resultados y la matriz de confusión.
    """)
    return


if __name__ == "__main__":
    app.run()
