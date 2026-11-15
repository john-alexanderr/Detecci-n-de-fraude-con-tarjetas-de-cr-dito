# /// script
# [tool.marimo.display]
# theme = "dark"
# ///
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

    Este notebook muestra cómo se entrena y elige el MLP, paso a paso. Al abrir la aplicación no se carga el CSV, no se entrena ni se evalúa. El botón puede pulsarse más de una vez: cada clic repite la búsqueda y evalúa de nuevo la prueba reservada, que solo contiene 99 fraudes. Para la presentación, usa los resultados ya registrados y no pulses el botón.

    ## Paso 1 — Separar el desarrollo de la prueba reservada

    La búsqueda usa solo el desarrollo agrupado —entrenamiento + validación, 227.845 filas con 393 fraudes—. La prueba externa (56.962 filas, 99 fraudes) queda fuera y se evalúa una sola vez al final.

    ## Paso 2 — Validar con folds agrupados

    `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=44)` reparte el desarrollo en tres folds. Cada grupo es un identificador calculado a partir de las variables de entrada; `groups_fit` y `groups_validation` se combinan en el mismo orden que `X` e `y` para que las filas con variables idénticas queden juntas. La prueba reservada no participa. Tres folds con 393 fraudes dan una estimación inicial práctica, no un resultado exhaustivo.

    ## Paso 3 — Comparar dos candidatos

    | Candidato | Capas ocultas | `alpha` | `learning_rate_init` |
    | --- | --- | ---: | ---: |
    | 1 | `(32,)` | `1e-3` | `1e-3` |
    | 2 | `(64, 32)` | `1e-4` | `1e-3` |

    El primer candidato es una red pequeña de una capa; el segundo prueba más capacidad con dos capas. También se comparan dos intensidades de regularización (`alpha`); la tasa de aprendizaje y el resto de la configuración se mantienen iguales para que la búsqueda sea acotada. Estos valores son candidatos iniciales, no una arquitectura demostrada como óptima.

    Ambos usan ReLU, Adam, tamaño de lote 256, `max_iter=100`, `early_stopping=False` y semilla 42. El pipeline es el mismo en los dos: `StandardScaler → SMOTE → MLPClassifier`, ajustado dentro de cada fold. En cada fold, el escalador y SMOTE se ajustan solo con el subconjunto de entrenamiento; el fold de validación conserva su prevalencia natural. La parada temprana está desactivada porque su validación interna saldría de los datos ya remuestreados por SMOTE; la selección se hace con la validación agrupada separada.

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

    ### Cómo leer la configuración ganadora

    | Parámetro o elemento | Qué significa en este modelo |
    | --- | --- |
    | `hidden_layer_sizes=(64, 32)` | Dos capas ocultas: la primera tiene 64 neuronas y la segunda 32. |
    | `activation="relu"` | ReLU se aplica en las capas ocultas para que la red pueda aprender relaciones no lineales. |
    | Salida binaria y pérdida | scikit-learn usa una salida logística para estimar la probabilidad de fraude y entrena con entropía cruzada (`log-loss`); no hace falta configurar esa salida por separado. |
    | `alpha=1e-4` | Regularización L2: penaliza pesos grandes. Se comparó `1e-4` con `1e-3`; el menor valor ganó por un margen mínimo en la validación, no porque se haya demostrado que evita el sobreajuste. |
    | `solver="adam"` | Optimizador que actualiza los pesos a partir del gradiente. Se mantuvo fijo; no se comparó con otros optimizadores. |
    | `batch_size=256` | Cantidad de ejemplos usados en cada actualización por lotes. |
    | `max_iter=100` | Límite de hasta 100 iteraciones de entrenamiento (épocas con Adam); no garantiza que el ajuste haya convergido. |
    | `early_stopping=False` | No crea una validación interna para detener el entrenamiento. En este pipeline esa validación se extraería de los datos ya remuestreados por SMOTE, no del fold separado con prevalencia natural. |

    ## Paso 4 — Elegir por Average Precision

    La selección usa el Average Precision medio de los tres folds. AP resume la curva precisión-recall con un promedio ponderado no interpolado: pondera la precisión por los cambios de recall y permite comparar el ranking de scores sin fijar un único umbral. No es el área geométrica trapezoidal exacta de la curva. Es una **búsqueda inicial acotada**, no un óptimo exhaustivo.

    ## Paso 5 — Reajustar y evaluar la prueba una sola vez

    `GridSearchCV(refit=True)` reajusta el pipeline ganador con todas las filas de desarrollo. Luego, la prueba externa se evalúa una única vez y nunca se usa para seleccionar parámetros ni ajustar un umbral. `predict_proba(x_test)[:, 1]` devuelve un score de probabilidad de fraude para cada transacción y se usa para Average Precision; `predict(x_test)` devuelve etiquetas 0/1 y se usa para precisión, recall, F2 y la matriz de confusión. En este MLP binario `predict` aplica el umbral predeterminado de 0,5, que no está aprobado por el negocio.

    El notebook entrega la prueba reservada al evaluador así:

    ```python
    mlp_test_result = evaluate_mlp_test(
        mlp_search_result.best_estimator,
        partitions.x_test,
        partitions.y_test,
    )
    ```

    `evaluate_mlp_test` devuelve las métricas de ranking y de clasificación, los recuentos de confusión y una vista previa. La prueba contiene 56.962 filas y solo 99 fraudes, así que sus métricas son una única estimación final sobre datos reservados, no una prueba del rendimiento en producción.
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
    mo.md(r"""
    ### Qué ejecuta el botón, en orden

    1. `load_dataset()` lee y valida el CSV; luego se seleccionan las 30 columnas de `FEATURE_COLUMNS` y se separa `Class` como objetivo.
    2. `create_grouped_partitions(features, target)` genera entrenamiento, validación y prueba reservada, junto con sus identificadores de grupo.
    3. `run_mlp_search(partitions)` combina solo entrenamiento y validación con sus grupos alineados. Prueba los dos candidatos con tres folds agrupados, devuelve los resúmenes de AP y parámetros, y `GridSearchCV(refit=True)` reajusta el ganador con todo el desarrollo.
    4. `evaluate_mlp_test(best_estimator, x_test, y_test)` aplica el pipeline ganador a la prueba reservada y devuelve métricas, recuentos de confusión y una vista previa.

    `mo.stop` mantiene esta secuencia detenida hasta el clic. Las tablas de validación cruzada y de prueba que aparecen abajo son resultados en vivo, generados tras ejecutar el botón; la interpretación escrita y el informe enlazado son registros guardados y no se recalculan ni actualizan al pulsarlo.
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
def _(mo):
    mo.md(r"""
    La vista previa contiene hasta diez filas de la prueba. Sus columnas son `actual_label` (clase real), `fraud_probability` (score de `predict_proba`) y `predicted_label` (clase de `predict`).
    """)
    return


@app.cell
def _(mlp_test_result):
    mlp_test_result.preview
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Guía de interpretación

    Cómo leer las métricas:

    - **Average Precision (AP)**: promedio ponderado no interpolado sobre la curva precisión-recall; resume el ranking de los scores y no es el área geométrica trapezoidal exacta.
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
