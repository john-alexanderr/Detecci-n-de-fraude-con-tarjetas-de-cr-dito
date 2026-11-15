<div align="center">

# Detección de Fraude con Tarjetas de Crédito

**Un benchmark académico y reproducible para detectar transacciones fraudulentas bajo desbalance extremo de clases.**

<p>
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn 1.9.1">
  <img src="https://img.shields.io/badge/imbalanced--learn-0.14.2-2C4F7C" alt="imbalanced-learn 0.14.2">
  <img src="https://img.shields.io/badge/Marimo-0.23.16-5B5BD6" alt="Marimo 0.23.16">
  <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"><img src="https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle Dataset"></a>
</p>

<p align="center">
  <a href="https://john-alexanderr.github.io/Detecci-n-de-fraude-con-tarjetas-de-cr-dito/"><strong>Artículo publicado →</strong> El juego del gato y el ratón: los grandes desafíos detrás de un sistema de detección de fraude</a>
</p>

</div>

---

## Qué se hizo con los datos

| Etapa | Descripción | Evidencia |
| :--- | :--- | :--- |
| **1. Auditoría y EDA** | Verificación de esquema, tipos, nulos, duplicados y rangos de las 30 variables. Análisis de prevalencia, `Amount`, componentes PCA y tiempo relativo. | [eda_summary.md](reports/eda_summary.md) |
| **2. Splits agrupados** | Las filas con entradas idénticas se mantienen juntas en una misma partición. Desarrollo: 227.845 filas (182.276 entrenamiento + 45.569 validación). Test reservado: 56.962 filas con 99 fraudes, nunca usado para ajustar, seleccionar ni tune. | [splitting.py](src/fraud_detection/splitting.py) |
| **3. Baselines de árboles** | `StandardScaler → SMOTE → clasificador` ajustado solo con el entrenamiento interno; la validación conserva la prevalencia natural. | [baseline_validation.md](reports/baseline_validation.md) |
| **4. MLP con CV agrupada** | Búsqueda de dos candidatos con `StratifiedGroupKFold` (3 folds, IDs de grupo) y `average_precision`. El ganador se reajustó con todos los datos de desarrollo y la prueba reservada se evaluó **una sola vez**. | [mlp_final_evaluation.md](reports/mlp_final_evaluation.md) |

### Resultados de los baselines (validación, 79 fraudes)

| Modelo | Average Precision | Precisión | Recall | F2 |
| :--- | ---: | ---: | ---: | ---: |
| **Random Forest** | **0.8499** | **0.8767** | 0.8101 | 0.8226 |
| AdaBoost | 0.7367 | 0.0482 | **0.9114** | 0.1988 |

### Resultados de la MLP

| Candidato | Media AP (CV) | Desv. est. |
| :--- | ---: | ---: |
| `(32,)` · `alpha=1e-3` · `lr=1e-3` | 0.7890 | 0.0163 |
| **`(64, 32)` · `alpha=1e-4` · `lr=1e-3` — elegido** | **0.7893** | 0.0248 |

Evaluación final sobre la prueba reservada (56.962 filas, 99 fraudes), sin ajuste de umbral:

| Average Precision | Precisión | Recall | F2 |
| ---: | ---: | ---: | ---: |
| **0.8204** | 0.7155 | 0.8384 | 0.8105 |

| Real \ Predicho | Legítima (0) | Fraude (1) |
| :--- | ---: | ---: |
| **Legítima (0)** | 56.830 | 33 |
| **Fraude (1)** | 16 | 83 |

> Los baselines se evaluaron solo en validación y no son directamente comparables con la MLP, que se evaluó en la prueba reservada. La estimación final descansa en apenas 99 fraudes: es evidencia incierta, no rendimiento de producción.

---

## Dataset

**Fuente:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), fijado por el código a `mlg-ulb/creditcardfraud/versions/3`.

| Dato | Valor |
| :--- | :--- |
| Transacciones | 284.807 (dos días de septiembre de 2013) |
| Fraudes | 492 (≈ 0,172 %) |
| Variables | 30 numéricas: `Time`, `Amount` y `V1`–`V28` (PCA anonimizado) |
| Objetivo | `Class` (1 = fraude, 0 = legítima) |

La primera ejecución del Notebook 01 descarga el CSV vía KaggleHub en `data/raw/creditcard.csv`. El dato crudo está ignorado por Git y **no debe commitearse**. KaggleHub puede requerir una cuenta de Kaggle; nunca guardar credenciales en el repositorio.

---

## Requisitos

- **Python 3.12 o superior**

| Paquete | Versión |
| :--- | :--- |
| `kagglehub` | 1.0.3 |
| `numpy` | 2.5.4 |
| `pandas` | 3.1.1 |
| `matplotlib` | 3.11.2 |
| `seaborn` | 0.13.2 |
| `scikit-learn` | 1.9.1 |
| `imbalanced-learn` | 0.14.2 |
| `marimo` (dev) | 0.23.16 |
| `pytest` (dev) | >=8, <10 |

Todo está declarado en [pyproject.toml](pyproject.toml) y los comandos en el [Makefile](Makefile).

---

## Replicación

```bash
git clone git@github.com:john-alexanderr/Detecci-n-de-fraude-con-tarjetas-de-cr-dito.git
cd Detecci-n-de-fraude-con-tarjetas-de-cr-dito

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[dev]'    # o: make install

python3 -m pytest                     # o: make test
```

Abrir las aplicaciones Marimo como archivos Python comunes:

```bash
marimo edit notebooks/01_dataset_understanding_eda.py    # make eda
marimo edit notebooks/02_data_preparation_baselines.py   # make baselines
marimo edit notebooks/03_mlp_training_evaluation.py
```

- **Notebook 01** descarga el CSV fijado si falta y ejecuta el análisis descriptivo.
- **Notebook 02** crea las particiones y entrena los baselines solo al presionar los botones.
- **Notebook 03** no hace nada hasta presionar **Ejecutar la búsqueda acotada del MLP y la prueba final**; la corrida completa tarda unos 15 minutos en CPU.

> Re-ejecutar el Notebook 03 vuelve a evaluar la prueba reservada. Hacerlo solo para verificación deliberada: la prueba no debe usarse para iterar.

---

## Estructura del proyecto

```text
docs/                 Marco de negocio, datos, arquitectura, metodología y model card
data/raw/             CSV local de Kaggle (ignorado); .gitkeep mantiene la carpeta
notebooks/            Tres apps Marimo: EDA, baselines y MLP
reports/              Evidencia de EDA, validación de baselines y evaluación MLP
src/fraud_detection/  Carga de datos, splits agrupados, pipelines, gráficos y métricas
tests/                Tests livianos con datos sintéticos; sin descargas ni entrenamiento
```

---

## Limitaciones

- El dataset cubre apenas **dos días de 2013**; la generalización temporal y actual es desconocida.
- El desbalance severo (492 fraudes en total) hace incierta toda estimación, especialmente el holdout de 99 fraudes.
- Los umbrales no se ajustaron a ninguna función de costo ni capacidad de revisión institucional; las métricas usan umbrales por defecto.
- No hay identificadores de tarjeta o cliente, comercio, dispositivo ni calendario: no son posibles análisis de historial, velocidad de transacciones ni estacionalidad.

---

## Documentación

- [Requerimientos de negocio](docs/business_requirements.md)
- [Diccionario de datos](docs/data_dictionary.md)
- [Arquitectura](docs/architecture.md)
- [Metodología](docs/methodology.md)
- [Model card](docs/model_card.md)
