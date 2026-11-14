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
    # 02 — Preparación de datos y modelos base

    Esta aplicación conserva los duplicados exactos de variables, pero mantiene cada grupo de variables idénticas dentro de una sola partición. Un botón crea la partición externa reservada agrupada y la división de validación agrupada; otro botón inicia el ajuste de los modelos. **Abrir la aplicación no divide el dataset completo, no ajusta ningún modelo y no ejecuta SMOTE.**

    ## ¿Por qué agrupar las variables de entrada exactas?

    Una división convencional de filas con estratificación puede colocar filas con variables idénticas en ambos lados. En la división convencional preservada, 227,845 filas de entrenamiento y 56,962 filas de prueba contenían 394 y 98 fraudes, respectivamente, pero 293 combinaciones exactas de variables cruzaban la división. En cambio, este benchmark calcula un hash solo de `X` y luego usa `StratifiedGroupKFold` para aproximar el equilibrio de clases, asignando cada grupo de variables por completo a una sola partición. Esta es una comparación de aprendizaje, no evidencia de generalización a producción ni a tiempos futuros.
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
    partition_run_button = mo.ui.run_button(label="Crear particiones agrupadas")
    partition_run_button
    return (partition_run_button,)


@app.cell
def _(create_grouped_partitions, features, mo, partition_run_button, target):
    mo.stop(
        not partition_run_button.value,
        mo.md(
            "Hacer clic en el botón de arriba para preparar las particiones agrupadas "
            "de entrenamiento, validación y prueba reservada."
        ),
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
        index=["entrenamiento del modelo", "validación", "prueba externa reservada"],
    )
    partition_summary
    return (partition_summary,)


@app.cell
def _(mo):
    outer_test_note = (
        "Los recuentos de clases de la prueba externa solo se muestran para verificar "
        "la estratificación. La partición de prueba no se usa para ajustar el "
        "preprocesamiento, seleccionar variables, modelos o umbrales, ni para "
        "evaluar modelos."
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
    mo.md(
        "Recuentos de superposición de grupos de variables exactas: "
        f"`{group_overlap_summary}`"
    )
    return (group_overlap_summary,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Preprocesamiento sin fuga de información

    El orden del pipeline base es **`StandardScaler -> SMOTE(random_state=42) -> classifier`**. Tanto el escalado como el remuestreo se ajustan solo con la partición de entrenamiento del modelo; la validación y la prueba externa reservada mantienen la prevalencia natural. SMOTE se basa en distancias, por lo que se estandarizan las 30 variables de entrada numéricas. Los árboles no necesitan escalado por sí mismos, pero usan el mismo pipeline para mantener coherente el límite de preprocesamiento de esta comparación.

    La configuración de los clasificadores coincide con el benchmark preservado: Decision Tree (`random_state=42`), Random Forest (`n_estimators=200`, `random_state=42`, `n_jobs=-1`) y AdaBoost (`random_state=42`).
    """)
    return


@app.cell
def _(mo):
    baseline_run_button = mo.ui.run_button(
        label="Entrenar los modelos base y evaluar la validación"
    )
    baseline_run_button
    return (baseline_run_button,)


@app.cell
def _(baseline_run_button, mo, partitions, pd, run_baseline_validation):
    mo.stop(
        not baseline_run_button.value,
        mo.md(
            "Hacer clic en el botón de arriba para ajustar los tres modelos base "
            "y evaluar la validación."
        ),
    )
    baseline_results, confusion_matrices = run_baseline_validation(
        partitions.x_fit,
        partitions.y_fit,
        partitions.x_validation,
        partitions.y_validation,
    )
    for model_name, matrix in confusion_matrices.items():
        print(
            f"Matriz de confusión de {model_name} "
            "(filas = real, columnas = predicho; etiquetas 0, 1):"
        )
        print(pd.DataFrame(matrix, index=[0, 1], columns=[0, 1]))
    baseline_results
    return baseline_results, confusion_matrices


@app.cell
def _(mo):
    mo.md(r"""
    ## MLP: completado en el Notebook 03

    El MLP ya se entrenó y evaluó en el [Notebook 03](03_mlp_training_evaluation.py): comparó dos candidatos con validación cruzada agrupada y evaluó una sola vez la prueba reservada. El candidato ganador usa capas ocultas `(64, 32)`, ReLU, log-loss binaria, Adam con `learning_rate_init=1e-3`, `alpha=1e-4`, tamaño de lote 256, `max_iter=100` y semilla 42. La parada temprana permanece desactivada dentro del pipeline posterior a SMOTE porque la validación corresponde a la partición separada con prevalencia natural. Los resultados exactos están en el [informe del MLP](../reports/mlp_final_evaluation.md).

    ## Límite de interpretación

    La comparación preservada tiene una sola división de validación con 79 fraudes. Random Forest logra el mejor equilibrio provisional en esa división; AdaBoost tiene el recall (exhaustividad) más alto, pero una carga severa de falsas alertas. Para estos baselines no existe validación cruzada, ajuste de umbral ni ajuste de hiperparámetros; la prueba reservada se evaluó solo con el MLP. Consultar [el informe exacto de validación](../reports/baseline_validation.md).
    """)
    return


if __name__ == "__main__":
    app.run()
