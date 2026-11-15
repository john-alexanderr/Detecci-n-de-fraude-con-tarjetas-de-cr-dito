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
    from fraud_detection.evaluation import run_baseline_validation
    from fraud_detection.models import build_baseline_pipelines
    from fraud_detection.plots import (
        plot_adaboost_concept,
        plot_confusion_matrix_concept,
        plot_confusion_matrices,
        plot_per_class_metrics,
        plot_random_forest_concept,
        set_plot_theme,
    )
    from fraud_detection.splitting import (
        create_grouped_partitions,
        grouped_overlap_count,
    )
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler

    # Marimo cambia la interfaz; esta llamada cambia las figuras Matplotlib/Seaborn.
    # Usa "light" si necesitas fondo claro.
    set_plot_theme("dark")

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
        plot_adaboost_concept,
        plot_confusion_matrices,
        plot_confusion_matrix_concept,
        plot_per_class_metrics,
        plot_random_forest_concept,
        run_baseline_validation,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 02 — Preparación de datos y modelos base

    Este notebook arma el modelo paso a paso: carga los datos, agrupa los duplicados exactos antes de dividir, construye el pipeline de preprocesamiento y entrena los dos modelos de referencia. Al abrir la aplicación se carga el CSV y se muestran las dimensiones de `X` e `y` junto con las primeras filas de `X`; los botones habilitan la división agrupada y el ajuste/evaluación de los modelos. Si falta `data/raw/creditcard.csv`, la carga intenta descargar la fuente fijada de Kaggle, por lo que abrir el notebook puede requerir conexión y descargar el archivo.

    > **Apariencia:** el tema de Marimo cambia la interfaz; `set_plot_theme("dark")` cambia los gráficos Matplotlib/Seaborn. Usa `set_plot_theme("light")` si necesitas un fondo claro.

    ## Paso 1 — Definir qué mira el modelo (`X`) y qué debe predecir (`y`)

    - **`X` (variables independientes)**: las 30 columnas numéricas que el modelo puede usar para decidir —`V1`–`V28`, `Time` y `Amount`—.
    - **`y` (variable dependiente)**: `Class`, la respuesta que queremos predecir (0 = legítima, 1 = fraude).

    Todavía no se transforma nada: la tabla original queda intacta. Este benchmark conserva `Time` para mantenerse fiel a la configuración preservada; por eso `X` tiene 30 columnas.

    Al abrir, `load_dataset()` lee y valida el esquema; `features` selecciona las columnas de `FEATURE_COLUMNS` y `target` separa `Class`. La salida de dimensiones cuenta filas y columnas de `X` y etiquetas de `y`; `features.head()` muestra las primeras cinco filas de entradas, no modifica los datos y no es una vista previa de las etiquetas.
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
    - **Prueba externa reservada** (56.962 filas, 99 fraudes): no se usa para elegir nada; el MLP la evalúa una sola vez al final. Los dos modelos base se comparan en validación.

    **La proporción 80/20.** Un `train_test_split` convencional con `test_size=0.2` separa 227.845 filas para entrenamiento (80 %) y 56.962 para prueba (20 %). Aquí la proporción es la misma, pero se respeta al agrupar: el 80 % de desarrollo se reparte en 182.276 filas para ajustar el modelo (64 % del total) y 45.569 para validar (16 % del total), y el 20 % restante queda como prueba externa. La prueba externa nunca participa de la validación cruzada; la columna `share_of_total_pct` de la tabla lo muestra.

    `mo.stop` detiene esta celda y las que dependen de las particiones mientras el botón no se haya pulsado; no impide que se carguen los datos ni que se muestre la vista previa anterior.
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
        "la estratificación. Esta partición no se usa para ajustar el preprocesamiento "
        "ni para seleccionar variables, modelos o umbrales. Los dos modelos base se "
        "evalúan en validación; el MLP (Notebook 03) se evalúa en la prueba una sola vez."
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
    Cada valor de la salida cuenta identificadores de grupo compartidos por un par de particiones. El `assert` detiene el flujo si alguno no es cero; tres ceros confirman que las filas con variables de entrada idénticas no se repartieron entre entrenamiento, validación y prueba.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Paso 3 — Construir el pipeline sin fuga de información

    **¿Por qué inspeccionar las escalas?** Las variables tienen magnitudes distintas. La tabla siguiente muestra la desviación estándar de cada columna en `partitions.x_fit`, solo como diagnóstico; no transforma los datos ni ajusta el escalador. `StandardScaler` aplica `z = (x − media) / desvío` usando estadísticas calculadas en entrenamiento.
    """)
    return


@app.cell
def _(partitions):
    scale_table = partitions.x_fit.std().sort_values(ascending=False).to_frame("std")
    scale_table.head(10)
    return (scale_table,)


@app.cell
def _(mo):
    mo.md(r"""
    La tabla ordena las diez mayores desviaciones observadas en entrenamiento y permite comparar las escalas sin usar estadísticas de validación o prueba. No es el escalador: cada pipeline calcula su propio `StandardScaler` al ajustarse con los datos de entrenamiento y reutiliza esas estadísticas para transformar los datos posteriores.

    El pipeline tiene tres pasos, siempre en este orden:

    1. **`StandardScaler`**: centra y escala las 30 variables con las estadísticas de entrenamiento. Esto hace comparables sus escalas para las distancias de SMOTE y la optimización del MLP.
    2. **`SMOTE(random_state=42)`**: crea fraudes sintéticos interpolando entre vecinos cercanos de la clase minoritaria. Se ajusta solo con el entrenamiento; la validación conserva la prevalencia natural (0,17 %).
    3. **Clasificador**: Random Forest (`n_estimators=200`, `random_state=42`, `n_jobs=-1`) y AdaBoost (`random_state=42`).

    Cada modelo tiene su propio pipeline y lo ajusta **solo** con la partición de entrenamiento. Ese orden evita que las estadísticas del escalador o los ejemplos sintéticos incorporen información de validación o prueba. Ambas particiones conservan su prevalencia natural.

    En cada pipeline, `fit` calcula el escalador con entrenamiento; validación y prueba solo se transforman con esas estadísticas. Ajustar otro escalador con la prueba también filtraría al proceso la distribución de datos reservados.

    **¿Qué hace SMOTE?** No copia fraudes: para cada fraude del entrenamiento busca sus vecinos más cercanos —en el espacio ya escalado— e inventa puntos intermedios entre ellos. Así la clase minoritaria crece hasta igualar a la mayoritaria. Se aplica **solo al entrenamiento**; validación y prueba conservan la prevalencia real (0,17 %) porque deben representar la realidad.

    La salida siguiente es la representación de la configuración del pipeline de Random Forest: muestra sus pasos y parámetros, pero no entrena el modelo.
    """)
    return


@app.cell
def _(build_baseline_pipelines):
    build_baseline_pipelines()["Random Forest"]
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Demostración independiente de SMOTE.** La tabla compara las clases antes y después de aplicar un escalador y una instancia demostrativa de SMOTE al subconjunto de entrenamiento: no reutiliza un objeto SMOTE ajustado dentro de los modelos. Random Forest y AdaBoost ajustan cada uno su propio escalador y sampler dentro de su pipeline. El equilibrio de la demostración queda en 181.962 filas por clase; validación y prueba no se remuestrean.
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
    ## Paso 4 — ¿Qué es un bosque aleatorio?

    Un **árbol de decisión** es una cadena de preguntas sobre las variables —"¿el importe supera 120 €?", "¿V14 es menor que −3?"— que termina en una hoja con una predicción: fraude o legítima. Es fácil de leer, pero frágil: si cambian un poco los datos de entrenamiento, el árbol puede cambiar por completo.

    El **bosque aleatorio** (Random Forest) resuelve esa fragilidad plantando muchos árboles distintos y haciéndolos votar:

    1. **Muestras distintas**: cada árbol se entrena con una muestra aleatoria *con reemplazo* (bootstrap) del conjunto de entrenamiento. Ningún árbol ve exactamente los mismos datos.
    2. **Variables distintas**: en cada corte, cada árbol solo puede elegir entre un subconjunto aleatorio de variables. Así los árboles no se parecen entre sí.
    3. **Votación**: cada árbol estima una probabilidad de fraude; el bosque promedia esas probabilidades y decide con un umbral (0,5 por defecto).
    4. **Resultado**: muchos árboles diversos compensan sus errores individuales. Eso es un *ensemble*: el conjunto es más robusto que cualquiera de sus partes.
    """)
    return


@app.cell
def _(plot_random_forest_concept):
    plot_random_forest_concept()
    return


@app.cell
def _(mo):
    mo.md(r"""
    En scikit-learn el clasificador se importa desde `sklearn.ensemble` y se entrena como cualquier modelo:

    ```python
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    forest.fit(partitions.x_fit, partitions.y_fit)
    ```

    En este proyecto el bosque vive dentro del pipeline —después del escalador y de SMOTE— con `n_estimators=200` y `random_state=42` para que el resultado sea reproducible. Los 200 árboles se entrenan en paralelo (`n_jobs=-1`), pero la decisión final sigue siendo un promedio.

    **AdaBoost, el otro modelo base**, sigue la idea opuesta: entrena árboles muy pequeños *en secuencia*, y cada uno se concentra en los casos que el anterior clasificó mal. Al final, los votos se ponderan por la calidad de cada árbol. Comparar ambos muestra dos formas distintas de combinar muchos árboles.

    **Ojo con el atajo habitual.** El ejemplo típico termina con `forest.score(test_X, test_y)`, que devuelve 0,9995: eso es exactitud, y con 0,17 % de fraudes no dice casi nada (Paso 6). La matriz que suele acompañar ese ejemplo —TN 56.849, FP 14, FN 10, TP 89— coincide con el tamaño de la partición de prueba de este proyecto (56.962 filas, 99 fraudes), pero aquí la prueba se reserva para el MLP y los modelos base se comparan en validación (Paso 8).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Paso 5 — ¿Qué es AdaBoost?

    El **impulso adaptativo** (Adaptive Boosting) también combina muchos árboles, pero de otra manera: **en secuencia**, no en paralelo.

    1. Se entrena un **tocón** (*stump*): un árbol de una sola pregunta, con dos hojas.
    2. Se miran los casos que falló y se les **sube el peso**. El siguiente tocón se concentra en esos casos difíciles.
    3. Se repite el proceso: cada tocón nuevo corrige en parte al anterior.
    4. La predicción final es una **votación ponderada**: cada tocón pesa según su acierto. Los que aciertan más, mandan más.

    Por eso se llama *adaptativo*: los pesos de las filas se adaptan en cada ronda a los errores del tocón anterior.
    """)
    return


@app.cell
def _(plot_adaboost_concept):
    plot_adaboost_concept()
    return


@app.cell
def _(mo):
    mo.md(r"""
    En scikit-learn se importa desde `sklearn.ensemble` y se entrena como cualquier modelo:

    ```python
    from sklearn.ensemble import AdaBoostClassifier

    boost = AdaBoostClassifier(random_state=42)
    boost.fit(partitions.x_fit, partitions.y_fit)
    ```

    `AdaBoostClassifier` usa por defecto tocones (`DecisionTreeClassifier(max_depth=1)`) y 50 rondas. En este proyecto vive dentro del pipeline, después del escalador y de SMOTE, con `random_state=42` para que sea reproducible.

    **Dos advertencias para la defensa.** Primero: que AdaBoost sea "resistente al sobreajuste" es una verdad a medias — ante ruido en las etiquetas es *sensible*, porque cada ronda sube el peso de los casos mal clasificados; si esos casos son fraudes ambiguos o etiquetas dudosas, el modelo insiste en ellos. Nuestro resultado en validación es la prueba: 1.423 falsas alarmas. Segundo: cuando se lee "el modelo tiene 97 % de precisión", casi siempre se habla de **exactitud**; la precisión de la clase fraude es otra cosa y aquí es 4,8 % (Paso 8).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Paso 6 — Entender cómo se evalúa: la matriz de confusión

    Esta parte es la brújula de todo lo que sigue. Si entiendes cómo se lee una matriz de confusión, el resto de las métricas se explica solo.

    ### 6.1 El modelo solo puede responder dos cosas

    Para cada transacción, el modelo elige entre dos respuestas: **legítima (0)** o **fraude (1)**. La realidad también tiene dos posibilidades. Al cruzar ambas aparece una tabla de 2 × 2 con exactamente cuatro resultados posibles. La convención de esta tabla es **filas = clase real**, **columnas = clase predicha**.
    """)
    return


@app.cell
def _(plot_confusion_matrix_concept):
    plot_confusion_matrix_concept()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.2 Las cuatro combinaciones posibles

    - **TP (verdadero positivo)**: era fraude y el modelo dijo fraude. Acierto.
    - **TN (verdadero negativo)**: era legítima y el modelo dijo legítima. Acierto.
    - **FP (falso positivo)**: era legítima y el modelo dijo fraude. Falsa alarma (error tipo I).
    - **FN (falso negativo)**: era fraude y el modelo dijo legítima. Fraude no detectado (error tipo II).

    Los aciertos están en la **diagonal** (TN y TP); los errores quedan fuera de ella (FP y FN).

    ### 6.3 Las cuatro cuentas que resumen todo

    | Métrica | Fórmula | Qué pregunta responde |
    | --- | --- | --- |
    | Exactitud (_accuracy_) | `(TP + TN) / total` | ¿Qué fracción de todas las respuestas fue correcta? |
    | Precisión | `TP / (TP + FP)` | De las alertas de fraude, ¿cuántas eran fraude real? |
    | Recall (exhaustividad) | `TP / (TP + FN)` | De todos los fraudes, ¿cuántos se detectaron? |
    | F1 | `2 · (precisión · recall) / (precisión + recall)` | Un solo número que equilibra precisión y recall |

    Precisión y recall tiran para lados opuestos: si el modelo grita "fraude" ante cualquier duda, sube el recall pero se llena de falsas alarmas (baja la precisión); si solo grita cuando está segurísimo, sube la precisión pero deja pasar fraudes (baja el recall).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.4 El mismo ejemplo con nuestros números

    En la validación hay **45.490 transacciones legítimas y 79 fraudes**. Random Forest produjo esta matriz:

    |  | Predijo legítima | Predijo fraude |
    | --- | --- | --- |
    | **Real legítima** | TN = 45.481 | FP = 9 |
    | **Real fraude** | FN = 15 | TP = 64 |

    Ahora las fórmulas dejan de ser símbolos y se convierten en cuentas:

    - **Precisión**: de las 73 alertas de fraude (64 + 9), 64 eran fraude real → `64 / 73 = 87,7 %`.
    - **Recall**: de los 79 fraudes reales (64 + 15), se detectaron 64 → `64 / 79 = 81,0 %`.
    - **Exactitud**: de las 45.569 transacciones, se clasificaron bien 45.545 (45.481 + 64) → `45.545 / 45.569 = 99,95 %`.

    Los 15 fraudes no detectados (FN) y las 9 falsas alarmas (FP) son tipos distintos de error. Su costo monetario u operativo no se puede cuantificar con este dataset.

    ### 6.5 ¿Y si el modelo no aprendiera nada?

    Un modelo que responda siempre "legítima" tendría `TN = 45.490`, `FP = 0`, `FN = 79`, `TP = 0`. Su exactitud sería `45.490 / 45.569 = 99,83 %`, casi idéntica a la de Random Forest (99,95 %), pero **no detecta un solo fraude**. Ahí se ve por qué la exactitud sola no alcanza cuando las clases están desbalanceadas.

    ### 6.6 Entonces, ¿qué miramos?

    La métrica principal es **Average Precision (AP)**: resume la curva precisión-recall como un promedio ponderado no interpolado y permite comparar el orden de los scores sin fijar un único umbral. No es el área geométrica trapezoidal exacta de la curva. Los informes también usan **F2**, que da al recall el doble de peso que a la precisión; es una elección de métrica, no una estimación de costos reales.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Paso 7 — Entrenar y comparar los dos modelos base

    Al pulsar el botón, `run_baseline_validation` ajusta cada pipeline con `x_fit` y `y_fit`, calcula en validación los scores de `predict_proba` para Average Precision y las etiquetas de `predict` para precisión, recall, F2 y matrices de confusión. No remuestrea la validación: sus 79 fraudes en 45.569 filas conservan la prevalencia natural. La función devuelve la tabla de esas métricas y las matrices; la columna de exactitud que se añade a la vista se calcula desde las matrices solo para explicar sus aciertos, no es una métrica de selección.

    Al terminar se muestran la tabla de métricas y las dos matrices de confusión con los recuentos reales. En cada matriz, la **diagonal** (TN y TP) son los aciertos; fuera de la diagonal quedan los errores (FP y FN).
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
    accuracy_by_model = {
        model_name: (matrix[0][0] + matrix[1][1])
        / sum(sum(row) for row in matrix)
        for model_name, matrix in confusion_matrices.items()
    }
    baseline_display_results = baseline_results.assign(
        accuracy=pd.Series(accuracy_by_model)
    )
    baseline_display_results
    return baseline_results, confusion_matrices


@app.cell
def _(confusion_matrices, plot_confusion_matrices):
    plot_confusion_matrices(confusion_matrices)
    return


@app.cell
def _(confusion_matrices, plot_per_class_metrics):
    plot_per_class_metrics(confusion_matrices)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Paso 8 — Leer los resultados

    La tabla traduce cada matriz a números. `accuracy` es una columna de presentación calculada a partir de TN, FP, FN y TP que devuelve la evaluación; no la calcula `run_baseline_validation` ni se usa para seleccionar un modelo. Observa de dónde sale cada porcentaje:

    | Métrica | Random Forest | AdaBoost | Cómo se obtiene |
    | --- | --- | --- | --- |
    | Exactitud | 99,95 % | 96,86 % | aciertos (TN + TP) / 45.569 |
    | Precisión | 87,67 % | 4,82 % | TP / (TP + FP) |
    | Recall | 81,01 % | 91,14 % | TP / (TP + FN) = TP / 79 |
    | F2 | 82,26 % | 19,88 % | combina precisión y recall, con peso doble al recall |

    Leer las dos matrices con la misma lupa:

    - **Random Forest**: 64 fraudes detectados (TP) y 15 no detectados (FN), con solo 9 falsas alarmas (FP). De cada 100 alertas que dispara, 88 son fraude real, y encuentra 81 de cada 100 fraudes. Es el mejor equilibrio provisional: mayor Average Precision y precisión.
    - **AdaBoost**: detecta 72 fraudes (TP) y deja escapar 7 (FN), pero dispara 1.423 falsas alarmas (FP). Encuentra 91 de cada 100 fraudes, pero de cada 100 alertas solo 5 son fraude real: su recall es más alto; su precisión, muy baja.

    ### Precisión y recall por clase

    La tabla anterior resume el modelo completo; esta mirada separa las dos clases. Es la que conviene tener a mano para explicar el comportamiento:

    - **Clase 0 (legítima)**: precisión y recall rondan el 100 %. No es un gran mérito: como el 99,8 % de las filas es legítima, acertar en esa clase es fácil y cualquier valor se ve casi perfecto.
    - **Clase 1 (fraude)**: aquí aparece la diferencia real. Random Forest acierta el 87,7 % de sus alertas y detecta el 81,0 % de los fraudes; AdaBoost detecta el 91,1 % pero solo el 4,8 % de sus alertas son fraude real.

    **Cuidado con los nombres al leer una matriz.** `TN` es la legítima bien clasificada y `TP` el fraude detectado: la celda grande (decenas de miles) es `TN`, no `TP`. Un texto que etiquete ese número como "verdadero positivo" está invirtiendo los términos, y con eso se invierten también FP y FN. En nuestra validación: `TN = 45.481`, `FP = 9`, `FN = 15`, `TP = 64`.

    ### Conclusión — qué muestran los resultados y qué no

    En esta partición de validación, Random Forest tiene mayor Average Precision, precisión y F2, y muchas menos falsas alarmas. AdaBoost detecta más fraudes, pero con una carga de alertas mucho mayor:

    | | Random Forest | AdaBoost |
    | --- | --- | --- |
    | Fraudes detectados | 64 de 79 (81,0 %) | 72 de 79 (91,1 %) |
    | Falsas alarmas | 9 | 1.423 |
    | Precisión de la clase fraude | 87,67 % | 4,82 % |
    | Average Precision | 0,850 | 0,737 |
    | Recall | 81,01 % | 91,14 % |
    | F2 | 0,823 | 0,199 |

    AdaBoost detecta **8 fraudes más** que Random Forest (72 frente a 64), con **1.414 falsas alarmas adicionales** en esta validación (1.423 frente a 9). Random Forest ofrece el mejor balance métrico provisional de estos dos baselines; AdaBoost obtiene mayor recall junto con ese volumen de alertas.

    **Conclusión condicionada:** si se priorizan AP, precisión, F2 y menos falsas alarmas en esta partición, Random Forest es el mejor balance provisional. Si se prioriza detectar más fraudes, AdaBoost alcanza mayor recall, pero su viabilidad depende de la capacidad real de revisión. No conocemos los costos institucionales de fraude y falsas alertas ni esa capacidad; además, `Amount` no equivale a la pérdida total. Por eso los resultados no demuestran que ninguno sea una opción operativa óptima para un banco. El MLP (Notebook 03) es el único modelo evaluado en la prueba reservada; esta comparación de baselines usa solo validación y no es directamente comparable con ese resultado.

    > **Para contarlo en 30 segundos:** en esta validación, Random Forest detecta 64 de 79 fraudes con 9 falsas alarmas; AdaBoost detecta 72, pero genera 1.423. Random Forest tiene mejores AP, precisión y F2; AdaBoost, mayor recall. Sin conocer los costos reales y la capacidad de revisión, podemos describir esa compensación, pero no declarar cuál sería la mejor decisión para un banco.

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
