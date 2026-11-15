# Arquitectura del proyecto

## Resumen

El repositorio es un pequeño benchmark académico, organizado en torno a tres aplicaciones Marimo y módulos de análisis reutilizables. Intencionalmente no contiene capas vacías de servicio, API, despliegue, monitoreo o streaming.

```text
docs/                 Marco del proyecto y metodología
data/raw/             Copia local del dataset (ignorada por Git)
notebooks/            Aplicaciones Marimo de EDA y de baseline agrupado
reports/              Resultados de análisis y validación conservados
src/fraud_detection/  Utilidades de dataset, particiones, modelos, gráficos y métricas
tests/                Pruebas unitarias con datos sintéticos
```

## Responsabilidades

| Componente | Responsabilidad |
| --- | --- |
| `fraud_detection.data` | Resuelve la ruta local del dataset dentro del repositorio, recupera la fuente fijada solo cuando no existe y valida su esquema. |
| `fraud_detection.splitting` | Calcula hashes de las firmas de variables de entrada y crea holdouts estratificados agrupados externos e internos. |
| `fraud_detection.models` | Construye los dos pipelines de baseline en un orden de preprocesamiento seguro frente a fugas de información. |
| `fraud_detection.plots` | Proporciona visualizaciones descriptivas reutilizables para la aplicación de EDA. |
| `fraud_detection.evaluation` | Ajusta y evalúa los baselines solo con datos de entrenamiento y validación. |
| `notebooks/01_dataset_understanding_eda.py` | Carga y describe la fuente, y luego muestra las vistas de EDA documentadas. |
| `notebooks/02_data_preparation_baselines.py` | Muestra las particiones agrupadas y ejecuta el ajuste de modelos solo después de una acción explícita en un botón. |

## Flujo de datos y límites

1. `data.raw.creditcard.csv` se reutiliza si existe; de lo contrario, KaggleHub recupera la versión 3 del dataset y solo se copia `creditcard.csv` de forma local.
2. La validación de esquema verifica las columnas esperadas, las entradas numéricas y el objetivo binario sin reescribir registros.
3. Las particiones externa e interna que respetan grupos usan solo hashes de `X`; los valores del objetivo no definen la pertenencia a un grupo.
4. Durante la acción opcional de baselines, el pipeline completo `StandardScaler -> SMOTE -> classifier` se ajusta con el subconjunto de entrenamiento. La validación se evalúa con la prevalencia natural.
5. La función auxiliar de evaluación no acepta datos de la prueba externa. Esa partición permanece sin usar.

## Configuración de dependencias y pruebas

`pyproject.toml` es la única fuente de configuración de dependencias y del paquete. Las dependencias de análisis fijadas son requisitos del proyecto; Marimo y pytest están en el extra `dev`. Las pruebas usan datos sintéticos pequeños y no descargan el archivo de Kaggle ni ajustan los modelos de baseline.
