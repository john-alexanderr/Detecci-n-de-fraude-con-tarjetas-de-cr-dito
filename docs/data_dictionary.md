# Diccionario de datos

## Origen y unidad de análisis

- **Origen:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), identificador del dataset `mlg-ulb/creditcardfraud/versions/3`.
- **Unidad:** Una transacción por fila.
- **Tarea:** Clasificación binaria de `Class`.
- **Alcance reportado del origen:** 284,807 transacciones de dos días de septiembre de 2013; 492 etiquetas de fraude.
- **Copia local:** `data/raw/creditcard.csv`, se descarga solo si no existe y Git la ignora.

## Columnas

| Columna(s) | Rol | Significado y limitaciones |
| --- | --- | --- |
| `Time` | Entrada numérica | Segundos transcurridos desde la primera transacción. No es una marca de tiempo calendario y no identifica la hora del reloj, el día de la semana ni la estacionalidad. |
| `V1`–`V28` | Entradas numéricas | Variables anonimizadas transformadas por PCA. No se proporciona su significado de negocio original. |
| `Amount` | Entrada numérica | Importe de la transacción; no es una medida completa de la pérdida por fraude ni del costo de negocio. |
| `Class` | Objetivo binario | `0` = legítima, `1` = fraude. |

Hay 30 entradas numéricas y un objetivo. El orden esperado de las columnas de origen es `Time`, `V1`–`V28`, `Amount`, `Class`.

## Hallazgos conocidos de calidad de datos

El perfil conservado no reporta valores faltantes, 1,081 filas duplicadas exactas y cero grupos de variables idénticas con objetivos en conflicto. Los posibles valores tipo centinela `Time = 999` (2 filas) y `Amount = 999` (7 filas) no están confirmados; no se deben recodificar automáticamente. Las marcas IQR indican valores para inspeccionar, no registros para eliminar o recortar. Consulte el [informe de EDA](../reports/eda_summary.md) para ver el detalle respaldado por scripts.
