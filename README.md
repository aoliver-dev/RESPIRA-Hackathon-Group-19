# Predicción de progresión de ILA mediante datos clínicos y proteómica

## Resumen

Este proyecto desarrolla un prototipo para estimar el riesgo de progresión de **anomalías pulmonares intersticiales (ILA)** a enfermedad pulmonar intersticial clínicamente significativa.

El objetivo no es diagnosticar EPID ya establecida, sino responder a una pregunta más temprana y clínicamente útil:

> **En un paciente que ya presenta ILA en el momento basal, ¿qué probabilidad tiene de progresar?**

Para ello combinamos información clínica disponible en la visita inicial con una firma sérica de cuatro proteínas.

---

## Cohorte utilizada

El dataset completo contiene 1.000 participantes distribuidos entre controles sanos, diferentes estados de ILA y EPID establecida.

Para el modelo predictivo principal utilizamos únicamente los pacientes que presentan ILA en el momento basal:

- 200 pacientes con ILA no fibrótica estable
- 150 pacientes con ILA fibrótica estable
- 300 pacientes con ILA progresiva

Total:

**650 pacientes con ILA**
- 300 progresores
- 350 no progresores

Los controles sanos y los pacientes con EPID ya establecida no se utilizan para entrenar el modelo principal porque harían el problema artificialmente más sencillo y no representan la pregunta clínica que queremos resolver.

---

## Variables utilizadas

### Variables clínicas basales

Se mantienen todas las variables clínicas de interés disponibles en el momento inicial:

- edad
- sexo
- IMC
- tabaquismo acumulado (pack-years)
- FVC basal
- DLCO basal
- patrón radiológico en TC

Aunque edad, sexo e IMC no mejoraron de forma relevante la discriminación en comparaciones preliminares, se conservaron para mantener un modelo clínicamente completo.

### Firma proteómica

La firma final está compuesta por:

- **SFTPB**
- **GDF15**
- **MMP7**
- **SFTPD**

Estas proteínas aportan señales complementarias relacionadas con lesión epitelial alveolar, respuesta al estrés celular y remodelado fibrótico.

---

## Variables excluidas

No se utilizan variables obtenidas durante el seguimiento, como:

- FVC de visitas posteriores
- delta FVC
- tiempos de visita posteriores

La razón es evitar **data leakage**: el modelo debe predecir el riesgo usando únicamente información disponible en el momento basal.

---

## Modelo de machine learning

Se utiliza una **regresión logística regularizada L2**.

La regresión logística combina todas las variables mediante coeficientes aprendidos automáticamente durante el entrenamiento y devuelve una probabilidad entre 0 y 1:

`P(progresión)`

La regularización L2 ayuda a evitar coeficientes excesivamente grandes y reduce el riesgo de sobreajuste.

---

## Preprocesado

Todo el preprocesado se realiza dentro del pipeline de entrenamiento para evitar que el conjunto de test aporte información al modelo.

### Variables numéricas

- imputación de valores ausentes mediante la mediana del training fold
- estandarización mediante media y desviación estándar del training fold

### Variables categóricas

- imputación por la categoría más frecuente del training fold
- one-hot encoding

---

## Validación

Se utiliza **validación cruzada estratificada de 5 folds** con semilla fija `2026`.

En cada iteración:

- aproximadamente 80% de los pacientes se utilizan para entrenamiento
- aproximadamente 20% se utilizan para evaluación

Cada paciente recibe una predicción de un modelo que no ha sido entrenado con ese paciente.

Estas predicciones se denominan **out-of-fold predictions** y son las utilizadas para calcular las métricas finales.

---

## Resultados principales

Modelo final: variables clínicas basales + SFTPB + GDF15 + MMP7 + SFTPD.

- **ROC-AUC: 0,9558**
- **Average Precision: 0,9378**
- **Brier Score: 0,0811**

El AUC de 0,9558 significa que, al escoger aleatoriamente un paciente progresor y otro no progresor, el modelo asigna un riesgo superior al progresor aproximadamente en el 95,6% de las parejas.

No significa que el modelo tenga una precisión del 95,6%.

---

## Estratificación de riesgo

A partir de las predicciones out-of-fold se definieron dos umbrales de desarrollo:

- **Bajo riesgo:** probabilidad < 0,279
- **Riesgo intermedio:** 0,279 ≤ probabilidad < 0,481
- **Alto riesgo:** probabilidad ≥ 0,481

Estos umbrales producen los siguientes grupos:

- **Bajo riesgo:** 4,8% de progresión observada
- **Intermedio:** 39,0%
- **Alto riesgo:** 88,6%

El umbral bajo se seleccionó buscando una sensibilidad de al menos el 95%.

El umbral alto se seleccionó buscando una especificidad de al menos el 90%.

Estos puntos de corte son únicamente de desarrollo y no constituyen umbrales clínicos validados.

---

## Estructura del repositorio

```text
fibrosis-pulmonar-hackathon/
├── README.md
├── requirements.txt
├── src/
│   ├── modelo_riesgo.py
│   └── calculos_resultados.py
├── app/
│   └── calculadora_riesgo.html
├── figuras/
├── resultados/
├── presentacion/
├── docs/
│   └── metodologia.md
└── data/
```

### `src/modelo_riesgo.py`

Contiene el pipeline del modelo predictivo y el ajuste de la regresión logística.

### `src/calculos_resultados.py`

Reproduce los cálculos utilizados para:

- comparar modelos
- obtener ROC-AUC
- calcular Average Precision
- calcular Brier Score
- obtener los umbrales de riesgo
- generar los grupos de riesgo
- exportar los CSV de resultados

### `app/calculadora_riesgo.html`

Prototipo interactivo que permite introducir los datos clínicos y proteómicos de un paciente y obtener una probabilidad estimada de progresión.

---

## Reproducibilidad

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Colocar el dataset en:

```text
data/reto_fibrosis_pulmonar_dataset_participantes.tsv
```

Ejecutar el cálculo de resultados:

```bash
python src/calculos_resultados.py
```

Los resultados se guardarán en la carpeta:

```text
resultados/
```

---

## Limitaciones

Este proyecto debe interpretarse como un prototipo de investigación.

Las principales limitaciones son:

- validación interna, no externa
- selección de la firma realizada sobre la misma cohorte de desarrollo
- ausencia de una cohorte independiente para confirmar generalización
- ausencia de metadatos de placa/lote para evaluar formalmente efectos técnicos residuales
- los umbrales de riesgo son de desarrollo y no deben utilizarse como puntos de corte clínicos

Antes de cualquier aplicación clínica sería necesario validar el modelo de forma prospectiva y externa.

---

## Mensaje principal

El resultado más importante no es únicamente un AUC elevado.

La propuesta combina:

**información clínica + imagen + biología molecular**

para transformar una ILA basal en una estimación individual de riesgo de progresión.

La firma final mantiene únicamente cuatro proteínas para favorecer interpretabilidad y potencial aplicabilidad clínica.
