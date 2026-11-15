import marimo

__generated_with = "0.23.16"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from fraud_detection.data import load_dataset, raw_data_path
    from fraud_detection.splitting import create_grouped_partitions
    from fraud_detection.plots import (
        plot_amount_by_class,
        plot_amount_by_transaction,
        plot_amount_distribution,
        plot_class_prevalence,
        plot_class_share,
        plot_correlation_heatmap,
        plot_iqr_flag_rates,
        plot_outlier_class_comparison,
        plot_pca_marginals,
        plot_relative_elapsed_time,
        plot_selected_pca_by_class,
        plot_target_associations,
        set_plot_theme,
    )

    set_plot_theme()
    return (
        load_dataset,
        mo,
        pd,
        create_grouped_partitions,
        plot_amount_by_class,
        plot_amount_by_transaction,
        plot_amount_distribution,
        plot_class_prevalence,
        plot_class_share,
        plot_correlation_heatmap,
        plot_iqr_flag_rates,
        plot_outlier_class_comparison,
        plot_pca_marginals,
        plot_relative_elapsed_time,
        plot_selected_pca_by_class,
        plot_target_associations,
        raw_data_path,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 01 — Comprensión del dataset y análisis exploratorio

    Construye una visión basada en evidencia del dataset público de fraude con tarjetas de crédito antes del modelado. Esta aplicación descarga la fuente fijada solo cuando el CSV local no existe y nunca edita ni elimina observaciones.

    ## Encuadre de negocio y límite de la evaluación

    Un equipo de pagos puede aprobar, rechazar, solicitar verificación adicional o revisar una transacción. No detectar un fraude puede aumentar las pérdidas; marcar como sospechosa una actividad legítima puede generar fricción con el cliente. El umbral adecuado depende de los costos y la capacidad de revisión de cada institución, datos que este dataset no proporciona.

    La fuente de Kaggle reporta **284,807 transacciones de dos días de septiembre de 2013**, incluidas **492 fraudes**. `Class` es la variable objetivo binaria (`0` legítima, `1` fraude). La exactitud por sí sola es engañosa con este desequilibrio tan severo; usa Average Precision / PR-AUC, precisión, recall (exhaustividad), F1 o F2 y matrices de confusión.

    Las celdas de fuente completa que aparecen a continuación se limitan al inventario de la fuente y a la prevalencia de etiquetas necesaria para la estratificación de la división; no son evidencia para seleccionar modelos. Después de hacer clic en un botón de forma explícita, cada gráfico e interpretación de EDA relevante para el modelo usa solo una partición de desarrollo agrupada (entrenamiento externo más validación). La prueba externa no se pasa a las funciones de EDA ni se resume en esta aplicación.

    ## Alcance del dataset

    Hay 30 variables de entrada numéricas: `Time`, `Amount` y `V1`–`V28`. `Time` son los segundos transcurridos desde la primera transacción, no una marca de tiempo de calendario. `V1`–`V28` son componentes PCA anonimizados y no tienen un significado de negocio directo suministrado. El dataset no incluye identificadores de tarjeta o cliente, campos de comercio, atributos demográficos ni secuencias de comportamiento.
    """)
    return


@app.cell
def _(load_dataset, mo, raw_data_path):
    data_path = raw_data_path()
    df = load_dataset(data_path)
    mo.md(
        f"Usando `{data_path.relative_to(data_path.parents[2])}`. "
        "El archivo de origen se reutiliza si existe; de lo contrario, solo se copia "
        "`creditcard.csv` del dataset de Kaggle, versión 3, en `data/raw/`."
    )
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Inventario de la fuente completa y diagnósticos de estratificación

    Las siguientes vistas de la fuente sin procesar describen el esquema, filas de ejemplo, estadísticas resumidas, valores faltantes, duplicados y prevalencia de etiquetas. Solo sirven para revisar la integridad de la entrada y la estratificación de clases; no son evidencia para seleccionar variables, transformaciones, modelos o umbrales.
    """)
    return


@app.cell
def _(df, pd):
    audit_overview = pd.DataFrame(
        {
            "Value": [
                len(df),
                df.shape[1],
                len(df.columns) - 1,
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
                int((df["Class"] == 1).sum()),
                float(df["Class"].mean() * 100),
            ]
        },
        index=[
            "Filas",
            "Columnas",
            "Variables de entrada numéricas",
            "Celdas faltantes",
            "Filas duplicadas exactas",
            "Filas de fraude",
            "Prevalencia de fraude (%)",
        ],
    )
    audit_overview
    return (audit_overview,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.dtypes.astype(str).rename("dtype").to_frame()
    return


@app.cell
def _(df):
    df.describe().T
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Estadísticas resumen del importe (`Amount`)

    `describe()` resume una columna numérica en ocho valores. Es una forma rápida de entender escala, dispersión y valores extremos antes de modelar:

    - `count`: filas con un valor presente (no cuenta nulos).
    - `mean`: promedio; se desplaza con facilidad si hay valores extremos.
    - `std`: desviación estándar; mide cuánto se dispersan los valores alrededor del promedio.
    - `min` y `max`: valores extremos observados.
    - `25%`, `50%` y `75%`: cuartiles. El `50%` es la mediana: la mitad de las transacciones queda por debajo.
    """)
    return


@app.cell
def _(df):
    df["Amount"].describe()
    return


@app.cell
def _(mo):
    mo.md(r"""
    La columna tiene 284,807 valores registrados, sin nulos. La media (`88.35`) es casi cuatro veces la mediana (`22.00`), y la desviación estándar (`250.12`) supera largamente a la media. Eso indica una distribución **muy sesgada a la derecha**: la mayoría de las transacciones son de importe bajo (el 50% no supera `22.00`), mientras unas pocas muy grandes empujan el promedio y el máximo (`25,691.16`). Por eso el análisis visual del importe usa la escala `log1p`, que comprime los valores grandes sin modificar los originales.
    """)
    return


@app.cell
def _(df):
    df["Class"].value_counts().rename("transactions").to_frame()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Interpretación del inventario de la fuente

    El perfil preservado de la fuente completa reporta 284,807 filas, 31 columnas, 1,081 filas duplicadas exactas (0.3796%) y cero grupos con variables de entrada idénticas y etiquetas contradictorias. No se detectaron valores faltantes. Los posibles valores tipo centinela son `Time = 999` (2 filas) y `Amount = 999` (7 filas); ninguno está confirmado como faltante. Estos diagnósticos pertenecen al inventario de la fuente y no son evidencia para seleccionar modelos. Conserva los duplicados y los valores inusuales durante esta auditoría descriptiva: estos hallazgos no autorizan a eliminar, recodificar ni limitar valores.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## EDA de modelado usando solo datos de desarrollo

    La creación de la partición agrupada está condicionada intencionalmente a un botón porque calcula un hash de toda la tabla de variables. La celda de división expone a las celdas de EDA solo la tabla combinada de desarrollo externo (entrenamiento más validación). No se muestran recuentos de la prueba externa ni resultados de EDA derivados de la prueba.
    """)
    return


@app.cell
def _(mo):
    development_eda_button = mo.ui.run_button(
        label="Crear partición de desarrollo agrupada"
    )
    development_eda_button
    return (development_eda_button,)


@app.cell
def _(create_grouped_partitions, df, development_eda_button, mo, pd):
    mo.stop(
        not development_eda_button.value,
        mo.md(
            "Hacer clic en el botón de arriba para preparar el EDA solo con datos "
            "de desarrollo."
        ),
    )
    grouped_partitions = create_grouped_partitions(
        df.drop(columns=["Class"]),
        df["Class"],
    )
    development_df = pd.concat(
        [
            grouped_partitions.x_fit.assign(Class=grouped_partitions.y_fit),
            grouped_partitions.x_validation.assign(
                Class=grouped_partitions.y_validation
            ),
        ]
    ).sort_index()
    return (development_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ### EDA parte 1 — Distribuciones univariadas

    Los gráficos usan solo la partición de desarrollo agrupada. Las diferencias de distribución por sí solas no demuestran valor predictivo.
    """)
    return


@app.cell
def _(development_df, plot_class_prevalence):
    plot_class_prevalence(development_df)
    return


@app.cell
def _(development_df, mo, plot_amount_distribution):
    mo.md(
        "El eje `log1p` solo comprime la visualización de la partición de desarrollo; "
        "los valores de origen no cambian."
    )
    plot_amount_distribution(development_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Dispersión del importe según la transacción

    Cada punto es una transacción: el eje horizontal muestra su importe real (sin `log1p`, para ver el rango completo) y el eje vertical su posición original en el archivo. El color indica la clase: azul intenso = 0 (legítima) y amarillo = 1 (fraude).

    La gran mayoría de los puntos son azules, así que la mancha amarilla de fraudes es pequeña y hay que mirarla de cerca. Además, no se ven fraudes por encima de 5.000 €: el importe máximo de una operación fraudulenta en todo el dataset es `2.125,87 €`, mientras que las legítimas llegan hasta `25.691,16 €`. Es una observación descriptiva, no una regla de decisión.
    """)
    return


@app.cell
def _(development_df, plot_amount_by_transaction):
    plot_amount_by_transaction(development_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Proporción entre clases

    No es de extrañar que apenas pudiéramos ver los puntos amarillos en el diagrama de dispersión de arriba, ya que solo el **0,173 %** de las transacciones son fraudulentas. Pero en absoluto es una cantidad insignificante.

    El gráfico de torta hace visible esa proporción. Por eso la exactitud (_accuracy_) sola no sirve como métrica: un modelo que clasifique todo como legítimo acertaría casi el 100 % y no detectaría ningún fraude.
    """)
    return


@app.cell
def _(development_df, plot_class_share):
    plot_class_share(development_df)
    return


@app.cell
def _(development_df, plot_pca_marginals):
    plot_pca_marginals(development_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### EDA parte 2 — Comparaciones bivariadas

    Los gráficos por clase y las asociaciones con el objetivo que aparecen a continuación usan solo la partición de desarrollo agrupada. Son evidencia exploratoria, no afirmaciones causales ni reglas automáticas de selección de variables.
    """)
    return


@app.cell
def _(development_df, plot_amount_by_class):
    plot_amount_by_class(development_df)
    return


@app.cell
def _(development_df, plot_selected_pca_by_class):
    plot_selected_pca_by_class(development_df)
    return


@app.cell
def _(development_df, mo, plot_target_associations):
    mo.md(
        "Las asociaciones de Pearson aquí se calculan solo sobre la partición de desarrollo; "
        "no son efectos causales ni prueba de fuga de información."
    )
    plot_target_associations(development_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### EDA parte 3 — Relaciones multivariadas

    La vista de correlación resume asociaciones lineales de a pares solo en la partición de desarrollo. Las correlaciones por sí solas no demuestran la utilidad de una variable ni justifican eliminarla.
    """)
    return


@app.cell
def _(development_df, plot_correlation_heatmap):
    plot_correlation_heatmap(development_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### EDA parte 4 — Patrón de tiempo transcurrido relativo

    Los intervalos de la partición de desarrollo que aparecen a continuación muestran recuentos y tasas de fraude a lo largo del tiempo transcurrido. `Time` no es tiempo de calendario, por lo que esto no permite sacar conclusiones sobre horas del día, días de la semana, feriados ni estacionalidad. La variación local tampoco permite establecer generalización temporal.
    """)
    return


@app.cell
def _(development_df, plot_relative_elapsed_time):
    plot_relative_elapsed_time(development_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### EDA parte 5 — Inspección de valores atípicos con IQR

    Los límites de 1.5 × IQR y las comparaciones por clase que aparecen a continuación se calculan solo sobre la partición de desarrollo. Marcan valores para revisión, pero no distinguen transacciones raras válidas de errores de datos. No se eliminan filas ni se limitan valores de forma automática.
    """)
    return


@app.cell
def _(development_df, plot_iqr_flag_rates):
    plot_iqr_flag_rates(development_df)
    return


@app.cell
def _(development_df, plot_outlier_class_comparison):
    plot_outlier_class_comparison(development_df)
    return


@app.cell
def _(development_df):
    classwise_outlier_quantiles = (
        development_df.groupby("Class")[["Amount", "V27"]].quantile(
            [0.5, 0.9, 0.99, 1.0]
        )
    )
    classwise_outlier_quantiles
    return (classwise_outlier_quantiles,)


if __name__ == "__main__":
    app.run()
