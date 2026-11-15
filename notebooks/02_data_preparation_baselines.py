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
    from fraud_detection.plots import (
        plot_confusion_matrix_concept,
        plot_confusion_matrices,
    )
    from fraud_detection.splitting import (
        create_grouped_partitions,
        grouped_overlap_count,
    )
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler

    return (
        FEATURE_COLUMNS,
        SMOTE,
        StandardScaler,
        build_baseline_pipelines,
        create_grouped_partitions,
        grouped_overlap_count,
        load_dataset,
        mo,
        pd,
        plot_confusion_matrices,
        plot_confusion_matrix_concept,
        run_baseline_validation,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 02 — Preparación de datos y modelos base

    Este notebook arma el modelo paso a paso: carga los datos, agrupa los duplicados exactos antes de dividir, construye el pipeline de preprocesamiento y entrena los dos modelos de referencia. **Abrir la aplicación no ejecuta nada**: la división y el entrenamiento se activan con botones.

    ## Paso 1 — Definir qué mira el modelo (`X`) y qué debe predecir (`y`)

    - **`X` (variables independientes)**: las 30 columnas numéricas que el modelo puede usar para decidir —`V1`–`V28`, `Time` y `Amount`—.
    - **`y` (variable dependiente)**: `Class`, la respuesta que queremos predecir (0 = legítima, 1 = fraude).

    Todavía no se transforma nada: la tabla original queda intacta. Este benchmark conserva `Time` para mantenerse fiel a la configuración preservada; por eso `X` tiene 30 columnas.
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
def _(features, target):
    _rows_text = f"{features.shape[0]:,}".replace(",", ".")
    print(
        f"X: {_rows_text} filas × {features.shape[1]} variables "
        f"· y: {_rows_text} etiquetas"
    )
    features.head()
    return


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

    **La proporción 80/20.** Un `train_test_split` convencional con `test_size=0.2` separa 227.845 filas para entrenamiento (80 %) y 56.962 para prueba (20 %). Aquí la proporción es la misma, pero se respeta al agrupar: el 80 % de desarrollo se reparte en 182.276 filas para ajustar el modelo (64 % del total) y 45.569 para validar (16 % del total), y el 20 % restante queda como prueba externa. La prueba externa nunca participa de la validación cruzada; la columna `share_of_total_pct` de la tabla lo muestra.
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
    partition_rows = [
        len(partitions.x_fit),
        len(partitions.x_validation),
        len(partitions.x_test),
    ]
    total_rows = sum(partition_rows)
    partition_summary = pd.DataFrame(
        {
            "rows": partition_rows,
            "share_of_total_pct": [
                100 * value / total_rows for value in partition_rows
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

    **¿Por qué escalar?** Las variables no están en la misma escala: `Time` supera los 47.000 y `Amount` los 250 de desvío, mientras que `V1`–`V28` rondan 1,5–2. Sin escalar, el algoritmo puede dar más peso a las columnas con números más grandes solo por su magnitud. `StandardScaler` aplica `z = (x − media) / desvío` y deja todas las variables con media 0 y desvío 1.
    """)
    return


@app.cell
def _(features):
    scale_table = features.std().sort_values(ascending=False).to_frame("std")
    scale_table.head(10)
    return (scale_table,)


@app.cell
def _(mo):
    mo.md(r"""
    La tabla lo confirma: los desvíos más grandes son los de `Time` y `Amount`; las componentes `V1`–`V28` están dos órdenes de magnitud por debajo. Después del escalado, todas las columnas pesan igual en distancia y en gradiente.

    El pipeline tiene tres pasos, siempre en este orden:

    1. **`StandardScaler`**: lleva las 30 variables a media 0 y desvío 1. SMOTE calcula distancias y un MLP es sensible a las escalas; estandarizar mantiene comparable el límite de preprocesamiento de la comparación.
    2. **`SMOTE(random_state=42)`**: crea fraudes sintéticos interpolando entre vecinos cercanos de la clase minoritaria. Se ajusta solo con el entrenamiento; la validación conserva la prevalencia natural (0,17 %).
    3. **Clasificador**: Random Forest (`n_estimators=200`, `random_state=42`, `n_jobs=-1`) y AdaBoost (`random_state=42`).

    Todo el pipeline se ajusta **dentro** de la partición de entrenamiento. Ese orden es lo que evita la fuga: si se escalara o se aplicara SMOTE antes de dividir, información de validación y de prueba se filtraría al modelo. La validación y la prueba externa reservada mantienen su prevalencia natural.

    **El escalador no se ajusta dos veces.** El `fit` se hace solo con entrenamiento; validación y prueba únicamente se transforman con ese escalador ya ajustado. Ajustar un escalador aparte con la prueba también sería una fuga: el modelo estaría usando la distribución de datos que nunca debe ver.

    **¿Qué hace SMOTE?** No copia fraudes: para cada fraude del entrenamiento busca sus vecinos más cercanos —en el espacio ya escalado— e inventa puntos intermedios entre ellos. Así la clase minoritaria crece hasta igualar a la mayoritaria. Se aplica **solo al entrenamiento**; validación y prueba conservan la prevalencia real (0,17 %) porque deben representar la realidad.

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
    **El efecto de SMOTE en este entrenamiento.** La tabla compara las clases antes y después de remuestrear la partición de entrenamiento del modelo (182.276 filas). El equilibrio queda en 181.962 de cada clase; la validación y la prueba no se tocan.
    """)
    return


@app.cell
def _(SMOTE, StandardScaler, partitions, pd):
    scaled_fit = StandardScaler().fit_transform(partitions.x_fit)
    _, smote_demo_target = SMOTE(random_state=42).fit_resample(
        scaled_fit, partitions.y_fit
    )
    smote_effect = pd.DataFrame(
        {
            "antes": partitions.y_fit.value_counts().sort_index().to_numpy(),
            "después": smote_demo_target.value_counts().sort_index().to_numpy(),
        },
        index=["legítima (0)", "fraude (1)"],
    )
    smote_effect
    return (smote_effect,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Paso 4 — Entender cómo se evalúa: la matriz de confusión

    Antes de entrenar conviene tener claro qué significa "predecir bien". La matriz de confusión compara lo que el modelo predijo con la etiqueta real. Convención de esta tabla: **filas = clase real**, **columnas = clase predicha**.

    Cada celda tiene un nombre:

    - **TP (verdadero positivo)**: era fraude y el modelo dijo fraude. Acierto.
    - **TN (verdadero negativo)**: era legítima y el modelo dijo legítima. Acierto.
    - **FP (falso positivo)**: era legítima y el modelo dijo fraude. Falsa alarma (error tipo I).
    - **FN (falso negativo)**: era fraude y el modelo dijo legítima. Fraude no detectado (error tipo II).

    De ahí salen las métricas:

    | Métrica | Fórmula | Qué responde |
    | --- | --- | --- |
    | Exactitud (_accuracy_) | `(TP + TN) / total` | ¿Qué fracción de todas las respuestas fue correcta? |
    | Precisión | `TP / (TP + FP)` | De las alertas de fraude, ¿cuántas eran fraude real? |
    | Recall (exhaustividad) | `TP / (TP + FN)` | De todos los fraudes, ¿cuántos se detectaron? |
    | F1 | `2 · (precisión · recall) / (precisión + recall)` | Equilibrio entre precisión y recall |

    En este problema la exactitud sola engaña: con 0,17 % de fraudes, un modelo que responda siempre "legítima" acierta el 99,8 % y no detecta nada. Por eso la métrica principal es el **área bajo la curva precisión-recall (AUPRC)**, que aquí llamamos `Average Precision`: resume qué tan bien el modelo ordena los fraudes por encima de las legítimas, sin depender de un umbral fijo. El informe usa además **F2** en lugar de F1: F2 da el doble de peso al recall, porque un fraude no detectado suele costar más que una falsa alarma.
    """)
    return


@app.cell
def _(plot_confusion_matrix_concept):
    plot_confusion_matrix_concept()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Paso 5 — Entrenar y comparar los dos modelos base

    El botón ajusta cada pipeline con las 182.276 filas de entrenamiento (SMOTE incluido dentro del pipeline) y predice la validación. La validación tiene prevalencia natural —79 fraudes en 45.569 filas—, así que las métricas reflejan el desbalance real, no uno inflado por el remuestreo. Al terminar se muestran la tabla de métricas y las dos matrices de confusión con los recuentos reales.
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
def _(baseline_run_button, mo, partitions, run_baseline_validation):
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
    baseline_results
    return baseline_results, confusion_matrices


@app.cell
def _(confusion_matrices, plot_confusion_matrices):
    plot_confusion_matrices(confusion_matrices)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Paso 6 — Leer los resultados

    Leer las dos matrices con la misma lupa:

    - **Random Forest**: 64 fraudes detectados (TP) y 15 no detectados (FN), con solo 9 falsas alarmas (FP). Es el mejor equilibrio provisional: mayor Average Precision y precisión.
    - **AdaBoost**: detecta 72 fraudes (TP) y deja escapar 7 (FN), pero dispara 1.423 falsas alarmas (FP). Su recall es más alto; su precisión, muy baja.

    La comparación tiene una sola división de validación con 79 fraudes. Para estos modelos base no existe validación cruzada, ajuste de umbral ni ajuste de hiperparámetros; la prueba reservada se evaluó solo con el MLP. Consultar [el informe exacto de validación](../reports/baseline_validation.md).
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
