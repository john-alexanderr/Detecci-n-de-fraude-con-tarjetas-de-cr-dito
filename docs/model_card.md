# Model card: baselines preliminares de fraude y evaluación completa del MLP

## Resumen

Esta model card describe una comparación académica histórica de Random Forest y AdaBoost, además de la validación cruzada completa del MLP, el entrenamiento y reajuste con datos de desarrollo y una evaluación final en la prueba externa. Los resultados de los árboles son evidencia solo de validación y ningún modelo está aprobado para tomar decisiones de pago.

## Uso previsto

- Docencia y comparación reproducible de clasificadores en un benchmark público.
- Estudio de las compensaciones entre precisión y recall (exhaustividad) bajo un desbalance de clases severo.

## Uso fuera de alcance

No se deben usar estos modelos ni resultados para aprobar, rechazar o tomar de otro modo decisiones de pago reales. No se han evaluado para patrones de fraude actuales, un emisor específico, latencia operativa, equidad, capacidad de revisión ni rendimiento en producción.

## Datos

La entrada es el dataset de fraude con tarjetas de crédito de Kaggle (versión 3), reportado como 284,807 transacciones de dos días de 2013, con 492 etiquetas de fraude. Las entradas son `Time`, `Amount` y los componentes PCA anonimizados `V1`–`V28`; `Class` es el objetivo. No hay campos de tarjeta, cliente, comercio, demografía ni secuencias de comportamiento. `Time` son segundos transcurridos, no tiempo calendario.

## Límite del EDA y de la selección

El inventario de la fuente completa en la aplicación de EDA se limita a diagnósticos de integridad de la entrada y a la prevalencia de etiquetas para la estratificación de las particiones. El [informe de EDA](../reports/eda_summary.md) conservado contiene valores exactos de una auditoría descriptiva de la fuente completa previa al modelo; esos resultados históricos no se usaron para seleccionar variables, transformaciones, modelos ni umbrales. Después de una acción explícita en el botón de ejecución, la aplicación actual calcula gráficos e interpretaciones relevantes para el modelo solo a partir de la partición agrupada de desarrollo (entrenamiento externo más validación). No pasa filas de la prueba externa a las funciones de EDA ni reporta recuentos de la prueba externa.

## Evaluación

Los resultados de los baselines usan una única partición interna de validación que respeta los grupos de duplicados, con 45,569 filas y 79 fraudes. Un escalador y SMOTE se ajustan dentro de cada pipeline de solo entrenamiento; los datos de validación conservan la prevalencia natural. Los clasificadores usan sus umbrales por defecto. Las métricas exactas y las matrices de confusión están en [el informe de validación](../reports/baseline_validation.md).

Random Forest tiene el mejor equilibrio provisional en esta única partición. AdaBoost tiene el recall más alto, pero también una gran carga de falsas alertas. Estos hallazgos no son estimaciones del rendimiento fuera de esta partición.

La fase completada del MLP combina solo los datos internos de entrenamiento y validación para una búsqueda de tres folds con `StratifiedGroupKFold`, pasando los identificadores de grupo de variables. Su pipeline `StandardScaler -> SMOTE -> MLPClassifier` mantiene el escalado y el muestreo dentro de cada fold. El candidato 2, `hidden_layer_sizes=(64, 32)`, `alpha=0.0001` y `learning_rate_init=0.001`, fue el ganador de la búsqueda; su Average Precision medio en CV fue `0.7892942255438676` (desviación estándar `0.024821041864665654`). La ventaja media sobre el candidato 1 es mínima en relación con la variación entre folds y no establece una superioridad decisiva de la arquitectura. El pipeline seleccionado se reajustó con todos los datos de desarrollo y luego la prueba externa reservada se evaluó una vez después de la selección. Sus métricas fueron Average Precision `0.8203794692697799`, precisión `0.7155172413793104`, recall `0.8383838383838383` y F2 `0.810546875`, sobre 56,962 filas que incluyen 99 fraudes. Average Precision usa probabilidades; la precisión, el recall, F2 y los recuentos de la matriz de confusión usan las predicciones por defecto. No hubo ajuste de umbral. Consultar el [informe del MLP](../reports/mlp_final_evaluation.md) para ver los resultados de ambos candidatos y la matriz de confusión.

Los modelos de árbol de baseline tienen resultados solo de validación y **no son directamente comparables** con las métricas del MLP en la prueba externa, porque se evaluaron en particiones distintas.

## Limitaciones y riesgos

- El desbalance de clases severo y solo 79 ejemplos de fraude en validación hacen inciertas las estimaciones.
- La estimación final del holdout del MLP incluye solo 99 fraudes y es incierta; no se debe usar para seguir iterando ni para seleccionar umbrales.
- El dataset cubre dos días de 2013; la generalización actual y temporal es desconocida.
- Los umbrales no se seleccionaron con una función de costo o una capacidad de revisión aprobadas por una institución.
- Los datos no permiten un análisis de equidad demográfica ni de comportamiento a nivel de tarjeta o cliente.
- La búsqueda del MLP es solo una grilla inicial de dos candidatos; incluso después de ejecutarla, no será un óptimo exhaustivo.
- Ninguna de las dos fases de modelos establece la preparación para producción ni el rendimiento frente al fraude actual. El resultado del MLP no implica rendimiento con patrones de fraude actuales ni más allá de este dataset de dos días de 2013.
- No se debe usar de nuevo la prueba externa reservada para iterar. Si se desea, validar con validación cruzada los baselines de árboles y diseñar una comparación justa de modelos usando datos de desarrollo; seleccionar cualquier umbral solo con datos de desarrollo o de validación.

## Búsqueda completada del MLP

Una búsqueda acotada comparó dos candidatos de MLP usando 30 entradas numéricas. Ambos usaron ReLU, Adam, `learning_rate_init=0.001`, tamaño de lote 256, `max_iter=100`, `early_stopping=False` y semilla 42. El ganador seleccionado de la búsqueda usó capas ocultas `(64, 32)` y `alpha=0.0001`; el pipeline seleccionado se reajustó con todos los datos de desarrollo antes de la única evaluación en la prueba externa. La parada temprana se desactivó dentro del pipeline posterior a SMOTE para que los folds agrupados explícitos siguieran siendo el límite de validación. La búsqueda de dos candidatos no es exhaustiva y su ganador no es prueba de superioridad de la arquitectura.
