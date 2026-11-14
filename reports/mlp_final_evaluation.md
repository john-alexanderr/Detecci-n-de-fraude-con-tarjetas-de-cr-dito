# Validación cruzada agrupada del MLP y evaluación final

La búsqueda acotada del MLP, el reajuste con datos de desarrollo y una evaluación final en la prueba externa reservada están completos. La prueba se evaluó una vez después de seleccionar el candidato; no se usó para ajustar ni iterar.

## Límite de la evaluación

Las particiones internas de entrenamiento y validación se combinaron como datos de desarrollo. La prueba externa se excluyó de la búsqueda. Una búsqueda de tres folds con `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=44)` recibió los identificadores de grupo de variables de desarrollo. `GridSearchCV` seleccionó por Average Precision y, con `refit=True`, ajustó el pipeline ganador con todas las filas de desarrollo antes de evaluar una vez la prueba reservada.

El pipeline fue `StandardScaler -> SMOTE(random_state=42) -> MLPClassifier`, con el escalado y SMOTE ajustados solo dentro de cada fold de entrenamiento de la CV. Ambos candidatos usaron ReLU, Adam, tamaño de lote 256, `max_iter=100`, `early_stopping=False` y estado aleatorio 42.

## Resultados de la validación cruzada

| Candidato | `hidden_layer_sizes` | `alpha` | `learning_rate_init` | Media del Average Precision en CV | Desv. estándar del Average Precision en CV | Folds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `(32,)` | `0.001` | `0.001` | `0.7889909900343617` | `0.016266031474164548` | 3 |
| 2 | `(64, 32)` | `0.0001` | `0.001` | `0.7892942255438676` | `0.024821041864665654` | 3 |

**Ganador seleccionado de la búsqueda:** `hidden_layer_sizes=(64, 32)`, `alpha=0.0001`, `learning_rate_init=0.001`.

El candidato 2 tiene la media de Average Precision en CV más alta, pero su ventaja es mínima en relación con la variación entre folds. Es el ganador de la búsqueda, no evidencia decisiva de que su arquitectura sea superior.

## Evaluación final en la prueba reservada

| Filas de prueba | Fraudes en prueba | Average Precision | Precisión | Recall | F2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 56962 | 99 | 0.8203794692697799 | 0.7155172413793104 | 0.8383838383838383 | 0.810546875 |

La prueba se evaluó una vez después de la selección. Average Precision usa `predict_proba`; la precisión, el recall, F2 y los recuentos de la matriz de confusión usan las etiquetas por defecto de `predict`. No hubo ajuste de umbral.

Matriz de confusión, con las etiquetas reales como filas y las etiquetas predichas `[0, 1]` como columnas:

| Real \ Predicho | 0 | 1 |
| --- | ---: | ---: |
| 0 | 56830 | 33 |
| 1 | 16 | 83 |

Estos son recuentos en el orden TN `56830`, FP `33`, FN `16`, TP `83`.

## Interpretación y limitaciones

- Las puntuaciones de CV son estimaciones con datos de desarrollo de una búsqueda acotada de dos candidatos, no un óptimo exhaustivo.
- El resultado de la prueba externa es una única estimación incierta de holdout basada en solo 99 fraudes. No establece el rendimiento frente al fraude actual, la preparación para producción ni el rendimiento más allá de este dataset de dos días de 2013.
- Los modelos de árbol de baseline tienen resultados solo de validación. **No son directamente comparables** con las métricas del MLP en la prueba externa, porque las particiones evaluadas son distintas.
- El notebook generó una vista previa de predicciones, pero las filas de predicción individuales no se reproducen en la documentación estática.

## Trabajo futuro

- Validar con validación cruzada y ajustar los baselines de árboles si se desea.
- Seleccionar cualquier umbral operativo usando solo datos de desarrollo o de validación.
- Diseñar una comparación justa que evalúe los modelos en las mismas particiones y con el mismo protocolo adecuado.
- No volver a usar la prueba externa reservada para iterar.
