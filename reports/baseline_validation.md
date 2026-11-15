# Resultados de validación de baselines

## Límite de la evaluación

Estos resultados conservados provienen de una única partición interna de validación agrupada que contiene **45,569 filas y 79 fraudes**. La validación mantiene la prevalencia natural de clases. Los pipelines aplican `StandardScaler -> SMOTE -> classifier` solo a los datos de entrenamiento. La precisión, el recall, F2 y las matrices de confusión usan el umbral de predicción por defecto de cada clasificador; Average Precision usa las probabilidades predichas.

La partición de prueba externa no se usó para selección ni evaluación. No se realizaron validación cruzada, ajuste de umbral, ajuste de hiperparámetros ni entrenamiento del MLP.

## Métricas

| Clasificador | Average Precision | Precisión | Recall | F2 |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.849946 | 0.876712 | 0.810127 | 0.822622 |
| AdaBoost | 0.736748 | 0.048161 | 0.911392 | 0.198785 |

## Matrices de confusión

Las filas son etiquetas reales y las columnas son predicciones, en el orden `[0, 1]`.

| Clasificador | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 45,481 | 9 | 15 | 64 |
| AdaBoost | 44,067 | 1,423 | 7 | 72 |

## Interpretación

Random Forest es el mejor equilibrio **provisional** en esta única partición de validación: tiene el Average Precision, la precisión y el F2 más altos entre ambos resultados con umbral por defecto. AdaBoost detecta la mayor cantidad de fraudes (72 de 79; recall 0.911392), pero produce 1,423 falsas alertas y una precisión de 0.048161.

Esta comparación no establece un modelo final, un umbral aprobado por el negocio, el rendimiento frente al fraude actual ni la preparación para producción. El pequeño recuento de fraudes en validación y el período de origen de dos días de 2013 limitan las afirmaciones de generalización.
