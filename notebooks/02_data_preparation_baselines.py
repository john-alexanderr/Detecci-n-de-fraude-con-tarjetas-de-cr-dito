import marimo

__generated_with = "0.23.16"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from fraud_detection.data import FEATURE_COLUMNS, load_dataset
    from fraud_detection.evaluation import run_baseline_validation
    from fraud_detection.models import build_baseline_pipelines
    from fraud_detection.splitting import (
        create_grouped_partitions,
        grouped_overlap_count,
    )

    return (
        FEATURE_COLUMNS,
        build_baseline_pipelines,
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

    Este notebook arma el modelo paso a paso: carga los datos, agrupa los duplicados exactos antes de dividir, construye el pipeline de preprocesamiento y entrena los dos modelos de referencia. **Abrir la aplicación no ejecuta nada**: la división y el entrenamiento se activan con botones.

    ## Paso 1 — Cargar las variables de entrada y la etiqueta

    Las 30 variables numéricas (`V1`–`V28`, `Time`, `Amount`) forman `X`; la columna `Class` (0 = legítima, 1 = fraude) es la etiqueta `y`. Todavía no se transforma nada: la tabla original queda intacta.
    """)
    return


@app.cell
def _(mo):
    mo.mermaid("""
    flowchart TD
        A["CSV: 284.807 transacciones"] --> B["X: 30 variables · y: Class"]
        B --> C["Hash de X: grupos de variables idénticas"]
        C --> D["StratifiedGroupKFold"]
        D --> E["Entrenamiento 182.276 filas"]
        D --> F["Validación 45.569 filas"]
        D --> G["Prueba externa reservada 56.962 filas"]
        E --> H["Pipeline: StandardScaler → SMOTE → clasificador"]
        H --> I["Random Forest"]
        H --> J["AdaBoost"]
        I --> K["Ajuste y evaluación en validación"]
        J --> K
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
    mo.md(r"""
    ## Paso 2 — Crear las particiones agrupadas

    **¿Por qué agrupar?** Una división convencional de filas puede colocar filas con las mismas 30 variables en entrenamiento y en prueba. El modelo entonces "reconoce" casos que ya vio y las métricas se inflan. Aquí se calcula un hash solo de `X` y se usa `StratifiedGroupKFold`: cada grupo de variables idénticas cae completo en una sola partición.

    En la división convencional preservada, 227.845 filas de entrenamiento y 56.962 de prueba contenían 394 y 98 fraudes, pero **293 combinaciones exactas cruzaban la división**. Con grupos, ese cruce debe ser cero (se verifica abajo).

    **¿Qué produce el botón?** Tres particiones con roles distintos:

    - **Entrenamiento** (182.276 filas, 314 fraudes): ajusta el pipeline.
    - **Validación** (45.569 filas, 79 fraudes): compara modelos con la prevalencia real.
    - **Prueba externa reservada** (56.962 filas, 99 fraudes): no se usa para elegir nada; solo el MLP la evalúa, una vez, al final.
    """)
    return


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
    ## Paso 3 — Construir el pipeline sin fuga de información

    El pipeline tiene tres pasos, siempre en este orden:

    1. **`StandardScaler`**: lleva las 30 variables a media 0 y desvío 1. SMOTE calcula distancias y un MLP es sensible a las escalas; estandarizar mantiene comparable el límite de preprocesamiento de la comparación.
    2. **`SMOTE(random_state=42)`**: crea fraudes sintéticos interpolando entre vecinos cercanos de la clase minoritaria. Se ajusta solo con el entrenamiento; la validación conserva la prevalencia natural (0,17 %).
    3. **Clasificador**: Random Forest (`n_estimators=200`, `random_state=42`, `n_jobs=-1`) y AdaBoost (`random_state=42`).

    Todo el pipeline se ajusta **dentro** de la partición de entrenamiento. Ese orden es lo que evita la fuga: si se escalara o se aplicara SMOTE antes de dividir, información de validación y de prueba se filtraría al modelo. La validación y la prueba externa reservada mantienen su prevalencia natural.

    Así se ve el pipeline construido:
    """)
    return


@app.cell
def _(build_baseline_pipelines):
    build_baseline_pipelines()["Random Forest"]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Paso 4 — Entrenar los dos modelos y evaluar en validación

    El botón ajusta cada pipeline con las 182.276 filas de entrenamiento (SMOTE incluido dentro del pipeline) y predice la validación. La validación tiene prevalencia natural —79 fraudes en 45.569 filas—, así que las métricas reflejan el desbalance real, no uno inflado por el remuestreo.
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
            "Hacer clic en el botón de arriba para ajustar los dos modelos base "
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
    ## Paso 5 — Leer los resultados

    Cómo leer las métricas que aparecen arriba:

    - **Average Precision (AP)**: resume la curva precisión-recall; más cerca de 1 significa mejor ranking de fraudes.
    - **Precisión**: de todas las alertas que marcó el modelo, qué fracción era fraude real.
    - **Recall (exhaustividad)**: de todos los fraudes reales, qué fracción fue detectada.
    - **F2**: combina precisión y recall dando el doble de peso al recall.
    - **Matriz de confusión**: TN y TP son aciertos, FP son falsas alarmas y FN son fraudes que se escaparon.

    La comparación tiene una sola división de validación con 79 fraudes. Random Forest logra el mejor equilibrio provisional en esa división; AdaBoost tiene el recall más alto, pero una carga severa de falsas alertas. Para estos modelos base no existe validación cruzada, ajuste de umbral ni ajuste de hiperparámetros; la prueba reservada se evaluó solo con el MLP. Consultar [el informe exacto de validación](../reports/baseline_validation.md).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Próximo paso — El MLP (Notebook 03)

    El MLP ya se entrenó y evaluó en el [Notebook 03](03_mlp_training_evaluation.py): comparó dos candidatos con validación cruzada agrupada y evaluó una sola vez la prueba reservada. El candidato ganador usa capas ocultas `(64, 32)`, ReLU, log-loss binaria, Adam con `learning_rate_init=1e-3`, `alpha=1e-4`, tamaño de lote 256, `max_iter=100` y semilla 42. La parada temprana permanece desactivada dentro del pipeline posterior a SMOTE porque la validación corresponde a la partición separada con prevalencia natural. Los resultados exactos están en el [informe del MLP](../reports/mlp_final_evaluation.md).
    """)
    return


if __name__ == "__main__":
    app.run()
