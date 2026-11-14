# Informe de EDA: dataset de fraude con tarjetas de crédito

> **Alcance:** Este informe conserva los valores exactos de la auditoría descriptiva de la fuente completa previa al modelo. Esos diagnósticos históricos se mantienen por trazabilidad y no son evidencia para seleccionar variables, transformaciones, modelos ni umbrales. La aplicación de EDA actual usa la fuente completa para el inventario y para crear la partición agrupada; después de una acción explícita de partición, todos los gráficos e interpretaciones relevantes para el modelo usan solo la partición agrupada de desarrollo. Aquí no se reportan ni evalúan métricas separadas de la prueba externa.

## Resumen ejecutivo

- El dataset contiene 284,807 filas y 31 columnas; conserve las 1,081 filas duplicadas exactas durante esta auditoría descriptiva, ya que no hay grupos de variables idénticas con objetivos en conflicto (`eda-profiler.shape={"columns":31,"rows":284807}`; `eda-profiler.duplicates.exact=1081`; `eda-profiler.duplicates.feature_conflicting_targets=0`).
- El fraude es una clase minoritaria severa: 492 fraudes (0.1727%) frente a 284,315 transacciones legítimas (99.8273%); priorice PR-AUC, recall y F1 y mantenga cualquier remuestreo o ponderación de clases dentro de los folds de entrenamiento (`eda-target-analysis.classes=[{"label":"0","count":284315,"pct":99.8273},{"label":"1","count":492,"pct":0.1727}]`; `eda-target-analysis.imbalance=severe`; `eda-target-analysis.recommendations=PR-AUC/recall/F1, stratified CV, train-fold-only resampling/weights`).
- No se detectaron valores faltantes, pero los valores tipo centinela son solo candidatos —no faltantes confirmados— y las marcas IQR justifican una revisión, no la eliminación ni el recorte automático de filas (`eda-missingness-audit.rows_with_missing_pct=0.0`; `eda-missingness-audit.sentinel_candidates.Time=999,count=2`; `eda-missingness-audit.sentinel_candidates.Amount=999,count=7`; `eda-missingness-audit.outliers_iqr.Amount=31904`).
- El escaneo de fugas recomienda k-fold estratificado; la deriva es desconocida sin un CSV de prueba separado, y el AUC moderado de la variable única V14 no es prueba de fuga (`eda-leakage-audit.cv_recommendation="stratified k-fold (StratifiedKFold)"`; `eda-leakage-audit.drift=[]`; `eda-leakage-audit.single_feature_auc.V14.auc=0.9492`).

## Salud del dataset

| Fase | Señal | Valor | Fuente | Acción |
| --- | --- | --- | --- | --- |
| Estructura | Duplicados exactos y objetivos en conflicto | 1,081 duplicados exactos; cero grupos de variables idénticas con objetivos en conflicto (`eda-profiler.duplicates.exact=1081`; `eda-profiler.duplicates.feature_conflicting_targets=0`) | `eda-profiler.duplicates` | Conservar las filas; los diagnósticos no respaldan la deduplicación. |
| Estructura | Objetivo casi constante | `Class.quasi_constant=true` (`eda-profiler.Class.quasi_constant=true`) | `eda-profiler.Class.quasi_constant` | Tratarlo como una señal de desbalance, no como motivo para eliminar el objetivo. |
| Objetivo | Desbalance de clases | 492 fraudes (0.1727%) y 284,315 transacciones legítimas (99.8273%); `imbalance=severe` (`eda-target-analysis.classes=[{"label":"0","count":284315,"pct":99.8273},{"label":"1","count":492,"pct":0.1727}]`; `eda-target-analysis.imbalance=severe`) | `eda-target-analysis.classes`; `eda-target-analysis.imbalance` | Usar PR-AUC, recall y F1; conservar la prevalencia natural en los datos de validación. |
| Datos faltantes | Nulos y posibles centinelas | Filas con valores faltantes: 0.0%; los valores candidatos `Time=999` (2) y `Amount=999` (7) no están confirmados (`eda-missingness-audit.rows_with_missing_pct=0.0`; `eda-missingness-audit.sentinel_candidates.Time=999,count=2`; `eda-missingness-audit.sentinel_candidates.Amount=999,count=7`) | `eda-missingness-audit.rows_with_missing_pct`; `eda-missingness-audit.sentinel_candidates` | Confirmar el significado de los centinelas con contexto de dominio antes de tratarlos como faltantes. |
| Señales | Asimetría y asociaciones con el objetivo | `Amount.skew=16.9776`; las asociaciones de Pearson más fuertes de la lista incluyen `V17=-0.3265` y `V14=-0.3025` (`eda-feature-signals.numeric.Amount.skew=16.9776`; `eda-feature-signals.numeric.V17.pearson_target=-0.3265`; `eda-feature-signals.numeric.V14.pearson_target=-0.3025`) | `eda-feature-signals.numeric` | Tratar las correlaciones como asociaciones, no como causalidad; validar las señales candidatas en particiones. |
| Fugas | Orden, grupos y deriva | `row_index_target_corr=-0.0117`; no hay claves de grupo ni columnas de fecha; la deriva es desconocida porque no se proporcionó un CSV de prueba separado (`eda-leakage-audit.row_index_target_corr=-0.0117`; `eda-leakage-audit.group_keys=[]`; `eda-leakage-audit.temporal.date_columns=[]`; `eda-leakage-audit.drift=[]`) | `eda-leakage-audit` | Mantener el holdout agrupado documentado en el Notebook 02; no afirmar que la deriva está ausente. |

## Hallazgos por fase

### Estructura

- El perfil reporta 284,807 filas, 31 columnas y una huella de memoria estimada de 70,632,136 bytes (`eda-profiler.shape={"columns":31,"rows":284807}`; `eda-profiler.memory_estimate_bytes=70632136`).
- Los duplicados exactos son 1,081 filas (0.3796%), con cero grupos de variables idénticas con objetivos en conflicto; `Class` es casi constante. Estos hallazgos describen los datos y no autorizan eliminar registros (`eda-profiler.duplicates.exact=1081`; `eda-profiler.duplicates.exact_pct=0.3796`; `eda-profiler.duplicates.feature_conflicting_targets=0`; `eda-profiler.Class.quasi_constant=true`).

### Objetivo

- La distribución de clases es 284,315 filas legítimas (99.8273%) y 492 filas de fraude (0.1727%); la clase diagnóstica es desbalance severo (`eda-target-analysis.classes=[{"label":"0","count":284315,"pct":99.8273},{"label":"1","count":492,"pct":0.1727}]`; `eda-target-analysis.minority_pct=0.1727`; `eda-target-analysis.imbalance=severe`).
- Las recomendaciones proporcionadas incluyen PR-AUC, recall y F1; validación cruzada estratificada; y remuestreo o pesos aplicados solo dentro de los folds de entrenamiento (`eda-target-analysis.recommendations=PR-AUC/recall/F1, stratified CV, train-fold-only resampling/weights`).

### Datos faltantes

- Ninguna fila contiene valores faltantes, todas las columnas tienen cero entradas faltantes y no se encontraron patrones de ausencia conjunta (`eda-missingness-audit.rows_with_missing_pct=0.0`; `eda-missingness-audit.columns[*].missing=0`; `eda-missingness-audit.patterns=[]`).
- Los posibles valores tipo centinela son `Time=999` en 2 filas y `Amount=999` en 7 filas; no son valores faltantes confirmados (`eda-missingness-audit.sentinel_candidates.Time=999,count=2`; `eda-missingness-audit.sentinel_candidates.Amount=999,count=7`).
- Los recuentos de marcas IQR incluyen `Amount=31904`, `V27=39163`, `V28=30342`, `V20=27770` y `V8=24134` (`eda-missingness-audit.outliers_iqr.Amount=31904`; `eda-missingness-audit.outliers_iqr.V27=39163`; `eda-missingness-audit.outliers_iqr.V28=30342`; `eda-missingness-audit.outliers_iqr.V20=27770`; `eda-missingness-audit.outliers_iqr.V8=24134`). Estas son marcas para inspección, no evidencia de registros inválidos; no recorte ni elimine observaciones durante esta auditoría.

### Señales

- `Amount` tiene una fuerte asimetría a la derecha (`Amount.skew=16.9776`), mientras que `Time` es casi simétrico (`Time.skew=-0.0356`) (`eda-feature-signals.numeric.Amount.skew=16.9776`; `eda-feature-signals.numeric.Time.skew=-0.0356`).
- Las asociaciones de Pearson con el objetivo más fuertes de la lista son `V17=-0.3265`, `V14=-0.3025`, `V12=-0.2606` y `V10=-0.2169` (`eda-feature-signals.numeric.V17.pearson_target=-0.3265`; `eda-feature-signals.numeric.V14.pearson_target=-0.3025`; `eda-feature-signals.numeric.V12.pearson_target=-0.2606`; `eda-feature-signals.numeric.V10.pearson_target=-0.2169`). La correlación es asociación, no causalidad.
- No se marcaron pares colineales (`eda-feature-signals.collinear_pairs=[]`).

### Fugas

- La correlación entre el índice de fila y el objetivo es -0.0117; no se identificaron claves de grupo ni columnas de fecha calendario (`eda-leakage-audit.row_index_target_corr=-0.0117`; `eda-leakage-audit.group_keys=[]`; `eda-leakage-audit.temporal.date_columns=[]`).
- La estrategia de validación cruzada recomendada es k-fold estratificado (`eda-leakage-audit.cv_recommendation="stratified k-fold (StratifiedKFold)"`). El Notebook 02 conserva por separado su holdout que respeta los grupos de duplicados; elija el próximo diseño de CV teniendo en cuenta tanto el equilibrio de clases como la agrupación de duplicados.
- El AUC de variable única más alto de la lista es el valor moderado de V14, 0.9492; por sí solo no es prueba de fuga (`eda-leakage-audit.single_feature_auc.V14.auc=0.9492`).
- La deriva es desconocida, no ausente: no se proporcionó un CSV de prueba separado, por lo que el resultado de deriva está vacío (`eda-leakage-audit.drift=[]`).

## Orientación para visualizaciones

| Gráfico | Propósito | Hallazgo | Notas |
| --- | --- | --- | --- |
| Gráfico de recuento de clases con eje de recuento logarítmico | Hacer visible la etiqueta minoritaria junto a la clase mayoritaria | `eda-target-analysis.classes=[{"label":"0","count":284315,"pct":99.8273},{"label":"1","count":492,"pct":0.1727}]` | Reportar PR-AUC y métricas específicas por clase; la exactitud por sí sola es insuficiente para este desbalance. |
| Distribución de `log1p(Amount)` y comparación por clase | Mostrar la forma de cola larga del importe y comparar las distribuciones por clase | `eda-feature-signals.numeric.Amount.skew=16.9776`; `eda-missingness-audit.outliers_iqr.Amount=31904` | Mantener la transformación como descriptiva; no recortar ni eliminar registros basándose en el gráfico. |
| Asociación de componentes PCA y gráficos por clase | Inspeccionar las distribuciones de los componentes anonimizados y las asociaciones candidatas con el objetivo | `eda-feature-signals.numeric.V17.pearson_target=-0.3265`; `eda-feature-signals.numeric.V14.pearson_target=-0.3025` | Las correlaciones no implican causalidad. |
| Recuentos relativos por hora transcurrida y paneles de tasa de fraude | Comparar volumen, recuentos de fraude y tasa a lo largo del tiempo transcurrido | `eda-leakage-audit.temporal.date_columns=[]` | `Time` no es tiempo calendario; no se respaldan conclusiones sobre hora calendario o día de la semana. |

## Preguntas abiertas y próximos pasos

- [ ] Confirmar si los valores candidatos `Time=999` y `Amount=999` tienen significados específicos del dominio antes de tratar cualquiera como faltante (`eda-missingness-audit.sentinel_candidates.Time=999,count=2`; `eda-missingness-audit.sentinel_candidates.Amount=999,count=7`).
- [ ] Diseñar una validación cruzada que conserve la recomendación de estratificación y tenga en cuenta la estructura de grupos duplicados documentada en el Notebook 02 (`eda-leakage-audit.cv_recommendation="stratified k-fold (StratifiedKFold)"`; `eda-leakage-audit.group_keys=[]`).
- [ ] En la próxima fase de modelado, comparar los baselines con validación cruzada y ajustar hiperparámetros y umbrales solo con particiones de entrenamiento/validación; mantener el remuestreo o la ponderación dentro de cada fold de entrenamiento (`eda-target-analysis.recommendations=PR-AUC/recall/F1, stratified CV, train-fold-only resampling/weights`).
- [ ] Obtener un dataset de prueba separado o de un período posterior apropiado antes de hacer afirmaciones sobre deriva; el estado actual de la deriva es desconocido (`eda-leakage-audit.drift=[]`).

## Fuentes

| Skill | Campo | Valor |
| --- | --- | --- |
| `eda-profiler` | `shape` | `{"columns":31,"rows":284807}` |
| `eda-profiler` | `memory_estimate_bytes` | `70632136` |
| `eda-profiler` | `duplicates.exact`; `duplicates.exact_pct`; `duplicates.feature_conflicting_targets` | `1081`; `0.3796`; `0` |
| `eda-profiler` | `Class.quasi_constant` | `true` |
| `eda-target-analysis` | `classes`; `minority_pct`; `imbalance` | `[{"label":"0","count":284315,"pct":99.8273},{"label":"1","count":492,"pct":0.1727}]`; `0.1727`; `severe` |
| `eda-target-analysis` | `recommendations` | PR-AUC/recall/F1; stratified CV; train-fold-only resampling/weights |
| `eda-missingness-audit` | `rows_with_missing_pct`; `columns[*].missing`; `patterns` | `0.0`; `0`; `[]` |
| `eda-missingness-audit` | `sentinel_candidates.Time`; `sentinel_candidates.Amount` | `value=999,count=2`; `value=999,count=7` |
| `eda-missingness-audit` | `outliers_iqr.Amount`; `outliers_iqr.V27`; `outliers_iqr.V28`; `outliers_iqr.V20`; `outliers_iqr.V8` | `31904`; `39163`; `30342`; `27770`; `24134` |
| `eda-feature-signals` | `numeric.Amount.skew`; `numeric.Time.skew` | `16.9776`; `-0.0356` |
| `eda-feature-signals` | `numeric.V17.pearson_target`; `numeric.V14.pearson_target`; `numeric.V12.pearson_target`; `numeric.V10.pearson_target` | `-0.3265`; `-0.3025`; `-0.2606`; `-0.2169` |
| `eda-feature-signals` | `collinear_pairs` | `[]` |
| `eda-leakage-audit` | `row_index_target_corr`; `group_keys`; `temporal.date_columns`; `cv_recommendation`; `drift` | `-0.0117`; `[]`; `[]`; `stratified k-fold (StratifiedKFold)`; `[]` (deriva desconocida porque no se proporcionó un CSV de prueba separado) |
| `eda-leakage-audit` | `single_feature_auc.V14.auc` | `0.9492` (moderado; no es prueba de fuga) |
