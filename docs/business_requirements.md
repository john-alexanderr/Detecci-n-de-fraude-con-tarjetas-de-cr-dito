# Requisitos de negocio y alcance del benchmark

## Problema

El benchmark académico estudia cómo las variables de las transacciones ordenan posibles fraudes bajo un desbalance de clases severo. En un flujo de pagos real, una transacción marcada podría rechazarse, enviarse a verificación adicional o revisarse. Dejar pasar un fraude puede aumentar las pérdidas y el trabajo de disputas; las falsas alertas pueden generar fricción con el cliente y pérdida de ingresos.

## Objetivos de evaluación

- Comparar clasificadores binarios con Average Precision (AP), precisión, recall (exhaustividad), F2 y recuentos de la matriz de confusión.
- Hacer visible la carga de falsas alertas junto con la detección de fraude.
- Mantener cada partición de validación y prueba en la prevalencia natural de clases del dataset.
- Evitar que firmas idénticas de variables de entrada crucen la partición externa o interna.
- Mantener intacta la partición de prueba externa hasta que las decisiones de modelo estén congeladas.

## Afirmaciones respaldadas

El trabajo actual respalda el análisis descriptivo de un dataset histórico público y una comparación preliminar de validación agrupada. No estima costos financieros específicos de una institución, no identifica un umbral operativo aprobado por el negocio ni establece una política de fraude desplegable.

El dataset puede respaldar más experimentos académicos sobre auditoría de esquema y calidad de datos, análisis de distribución de variables, clasificación de eventos raros, compensaciones de umbral y benchmarking del tiempo de inferencia del modelo por sí solo en un entorno local declarado. Una partición cronológica podría ofrecer una simulación histórica limitada después de verificar el orden de las filas, pero dos días de datos aportan evidencia débil sobre la generalización temporal.

## Fuera de alcance

El dataset no tiene identificadores de tarjeta ni de cliente, atributos de comercio, huellas de dispositivo, direcciones IP, geografía ni atributos demográficos. No puede respaldar el comportamiento por tarjeta o por cliente, el riesgo de comercio, las verificaciones geográficas, el análisis de tiempo calendario (hora del día, día de la semana, feriados o estacionalidad), el análisis de equidad entre grupos demográficos ni la medición de resultados de aprobación, rechazo, contracargo, recuperación, investigación y abandono de clientes. `Amount` por sí sola no es una medida completa de la pérdida por fraude y no puede establecer los costos de falsos positivos o falsos negativos de la institución.

Este repositorio no implementa una API de pagos, un servicio de decisión en producción, streaming en tiempo real, monitoreo ni infraestructura operativa. El dataset no puede validar un SLA de autorización de extremo a extremo, el rendimiento actual frente al fraude, un manejo confiable de la deriva de concepto (concept drift) ni la generalización a otros períodos, regiones, emisores o canales de transacción. Cubre solo dos días de 2013.
