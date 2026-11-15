# Metodología

## Auditoría del dataset y análisis exploratorio

La aplicación de EDA mantiene intacta la fuente completa descargada. Sus celdas de inventario sobre la fuente completa reportan esquema, filas de muestra, estadísticas resumidas, recuentos de nulos, duplicados exactos y prevalencia de etiquetas solo para revisar la integridad de la entrada y para la estratificación de las particiones; no son evidencia para seleccionar modelos. La partición agrupada se crea solo después de un clic explícito en un botón. Luego, la aplicación combina las filas agrupadas de entrenamiento y validación de modelos en un marco de desarrollo y pasa solo ese marco a los gráficos condicionados por clase, las asociaciones con el objetivo, las correlaciones, los límites de outliers y el resto del EDA relevante para el modelo. Las filas de la prueba externa no se pasan a esas funciones y la aplicación de EDA no reporta recuentos de la prueba externa.

El [informe de EDA](../reports/eda_summary.md) conservado sigue siendo una auditoría descriptiva exacta de la fuente completa, previa al modelo, incluidos sus estadísticas y asociaciones a nivel de variable registradas anteriormente. Esos valores históricos se conservan por trazabilidad; no se usan para ajustar transformaciones, seleccionar variables, elegir modelos o umbrales, ni estimar rendimiento. Los gráficos son descriptivos, no evidencia causal ni base para eliminar o recortar valores automáticamente. `Time` son segundos transcurridos relativos y `V1`–`V28` no tienen una interpretación de negocio directa.

## Particiones que respetan grupos

1. Calcular un hash solo de las variables de entrada de cada fila; no incluir `Class` en la clave de grupo.
2. Usar `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` y su primer fold para la partición externa de desarrollo/prueba.
3. Usar un segundo `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)` y su primer fold dentro de desarrollo para el entrenamiento y la validación del modelo.
4. Mantener los grupos de variables de entrada idénticas por completo dentro de una misma partición en ambos niveles. Conservar todas las filas; no deduplicar.

La partición externa actual tiene 227,845 filas de desarrollo (393 fraudes) y 56,962 filas de prueba externa (99 fraudes). La partición interna tiene 182,276 filas de entrenamiento del modelo (314 fraudes) y 45,569 filas de validación (79 fraudes). Estas particiones conservan aproximadamente la prevalencia natural y no tienen superposición de grupos de variables idénticas entre particiones.

A modo de comparación, la partición aleatoria estratificada convencional anterior tenía 227,845 filas de entrenamiento (394 fraudes) y 56,962 filas de prueba (98 fraudes), pero 293 firmas distintas de variables idénticas cruzaban la partición. Esa superposición es la razón por la que la evaluación conservada usa particiones agrupadas.

## Preprocesamiento y baselines

Cada pipeline de baseline aplica **`StandardScaler -> SMOTE(random_state=42) -> classifier`**. El escalador y el muestreador se ajustan solo con el subconjunto de entrenamiento; las muestras generadas nunca entran en validación. La validación se mantiene en la prevalencia natural. Los dos clasificadores y sus configuraciones son:

| Clasificador | Configuración |
| --- | --- |
| Random Forest | `n_estimators=200`, `random_state=42`, `n_jobs=-1` |
| AdaBoost | `random_state=42` |

Las métricas usan el umbral de predicción por defecto de cada clasificador. Average Precision se calcula a partir de las probabilidades de validación; la precisión, el recall, F2 y las matrices de confusión usan la salida de `predict`. Los resultados exactos de la validación anterior están documentados en [Validación de baselines](../reports/baseline_validation.md).

## Búsqueda acotada del MLP y evaluación final

La fase del MLP se implementa en el [Notebook 03](../notebooks/03_mlp_training_evaluation.py) y comparte las particiones agrupadas deterministas existentes. Combina solo `x_fit` más `x_validation`, sus objetivos y sus identificadores de grupo de variables en los datos de desarrollo. La prueba externa de 56,962 filas permanece reservada; nunca se incluye en la validación cruzada.

La búsqueda usa `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=44)` y pasa los identificadores de grupo de desarrollo a `GridSearchCV.fit`. Tres folds son una elección inicial acotada para los 393 fraudes de desarrollo, no una búsqueda exhaustiva ni una afirmación de optimalidad. Los dos candidatos son:

| Candidato | `hidden_layer_sizes` | `alpha` | `learning_rate_init` |
| --- | --- | ---: | ---: |
| 1 | `(32,)` | `1e-3` | `1e-3` |
| 2 | `(64, 32)` | `1e-4` | `1e-3` |

Ambos candidatos usan ReLU, Adam, tamaño de lote 256, `max_iter=100`, `early_stopping=False` y estado aleatorio 42. El orden del estimador es **`StandardScaler -> SMOTE(random_state=42) -> MLPClassifier`** dentro del estimador de validación cruzada. Por lo tanto, cada fold ajusta el escalado y SMOTE solo con su partición de entrenamiento; sus datos de validación mantienen la prevalencia natural. La búsqueda usa `average_precision`, `refit=True`, un trabajo, un trabajo predespachado y `error_score="raise"` para que los errores del estimador sean visibles. La media y la desviación estándar del Average Precision de los folds de los candidatos se reportan junto con el número de folds.

Después de la selección, `GridSearchCV(refit=True)` ajusta el pipeline elegido con todas las filas de desarrollo. La prueba externa reservada se evaluó una vez después de la selección: Average Precision usa `predict_proba`, mientras que la precisión, el recall, F2 y los recuentos TN/FP/FN/TP de la matriz de confusión usan las etiquetas por defecto de `predict`. No hubo ajuste de umbral. La prueba tiene 56,962 filas y 99 fraudes, por lo que su resultado es una única estimación incierta sobre un holdout, con evidencia limitada de la clase minoritaria; no es rendimiento de producción.

| Candidato | Media del Average Precision en CV | Desv. estándar del Average Precision en CV | Folds |
| --- | ---: | ---: | ---: |
| 1: `(32,)`, `alpha=0.001`, `learning_rate_init=0.001` | `0.7889909900343617` | `0.016266031474164548` | 3 |
| 2: `(64, 32)`, `alpha=0.0001`, `learning_rate_init=0.001` | `0.7892942255438676` | `0.024821041864665654` | 3 |

El candidato 2 fue el ganador de la búsqueda y se reajustó con todos los datos de desarrollo. Su ventaja media es mínima en relación con la variación entre folds, por lo que el resultado no demuestra una superioridad decisiva de la arquitectura. Los resultados finales en la prueba externa fueron Average Precision `0.8203794692697799`, precisión `0.7155172413793104`, recall `0.8383838383838383` y F2 `0.810546875`; los recuentos de la matriz de confusión fueron TN `56830`, FP `33`, FN `16` y TP `83`. La prueba se evaluó una vez y no hubo ajuste de umbral. Los resultados completos están en el [informe de resultados del MLP](../reports/mlp_final_evaluation.md).

Los baselines de árboles tienen resultados solo de validación y **no son directamente comparables** con las métricas del MLP en la prueba externa, porque las particiones evaluadas son distintas. La estimación del MLP en la prueba es incierta porque incluye solo 99 fraudes; no establece el rendimiento frente al fraude actual ni el rendimiento más allá de este dataset de dos días de 2013. No se debe usar de nuevo la prueba externa reservada para iterar.

## Límites y trabajo futuro

La comparación de baselines conservada es una sola partición de validación con 79 ejemplos de fraude y sigue siendo provisional. El experimento del MLP completado es una búsqueda inicial acotada, no un óptimo exhaustivo; su estimación final se basa en solo 99 fraudes de la prueba externa, y los datos subyacentes cubren dos días de 2013. Ni los resultados de los baselines ni los del MLP establecen el rendimiento frente al fraude actual, un umbral aprobado por el negocio ni la preparación para producción. El trabajo futuro puede incluir la validación cruzada de los baselines de árboles y el diseño de una comparación justa; cualquier selección de umbral debe usar solo datos de desarrollo o de validación.
