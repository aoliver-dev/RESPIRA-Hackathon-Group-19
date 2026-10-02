# Metodología

Este documento describe de forma reproducible cómo se construyó y evaluó el modelo de riesgo de progresión de ILA.

---

## 1. Pregunta clínica

El objetivo es predecir, en el momento basal, qué pacientes con anomalías pulmonares intersticiales (ILA) tienen mayor riesgo de progresar a enfermedad pulmonar intersticial clínicamente significativa.

Por tanto, el problema se formula como una clasificación binaria:

- `1` = ILA progresiva
- `0` = ILA estable

El modelo utiliza exclusivamente información disponible en la visita inicial.

---

## 2. Selección de la cohorte

Se excluyen del entrenamiento principal:

- controles sanos
- pacientes con EPID ya establecida

Se conservan:

- ILA no fibrótica estable
- ILA fibrótica estable
- ILA progresiva

Esto da una cohorte de:

- 300 progresores
- 350 no progresores
- 650 pacientes en total

La razón es evitar un problema de clasificación artificialmente sencillo. El objetivo no es distinguir sujetos sanos de pacientes con EPID, sino discriminar riesgo dentro de pacientes que ya presentan ILA.

---

## 3. Variable objetivo

Se crea una variable binaria:

```python
target = 1  # Progressive_ILA
target = 0  # Stable ILA
```

La regresión logística aprende a estimar:

`P(target = 1)`

Es decir, la probabilidad estimada de progresión.

---

## 4. Variables predictoras

### 4.1 Variables clínicas

- edad
- sexo
- IMC
- cigarette pack-years
- FVC basal
- DLCO basal
- patrón de TC

Edad, sexo e IMC se mantienen deliberadamente aunque su contribución adicional a la discriminación sea pequeña.

La razón es mantener un modelo clínicamente completo y evitar que la firma se interprete como puramente proteómica.

### 4.2 Variables proteómicas

Firma final:

- SFTPB
- GDF15
- MMP7
- SFTPD

La firma se seleccionó buscando un equilibrio entre:

1. capacidad predictiva
2. estabilidad
3. plausibilidad biológica
4. simplicidad

El modelo con cuatro proteínas mejora modestamente al modelo de dos proteínas y mantiene una firma pequeña e interpretable.

---

## 5. Variables excluidas para evitar leakage

No se utilizan variables posteriores al momento basal.

Ejemplos:

- FVC en visita 2
- FVC en visita 3
- delta FVC
- tiempo hasta visitas posteriores

Utilizar estas variables significaría proporcionar al modelo información que todavía no estaría disponible cuando se quiere hacer la predicción clínica.

---

## 6. Control de calidad proteómico

El dataset contiene 2.944 proteínas medidas en escala NPX.

Se evaluó:

- porcentaje de valores ausentes por proteína
- porcentaje de valores ausentes por paciente
- proteínas con missingness >20%
- proteínas de varianza nula o casi nula
- valores extremos
- estructura global mediante PCA

Resultados principales:

- missingness proteómico global: aproximadamente 1,99%
- ninguna proteína supera 20% de missingness
- ninguna proteína presenta varianza prácticamente nula
- no se detectaron muestras con missingness global preocupante

Por tanto, no fue necesario eliminar proteínas por criterios generales de calidad.

---

## 7. Imputación de valores ausentes

Para las variables numéricas se utiliza:

```python
SimpleImputer(strategy="median")
```

La mediana se calcula únicamente utilizando los pacientes del training fold.

Ejemplo:

si un fold contiene aproximadamente 520 pacientes de entrenamiento, se calcula la mediana de SFTPB entre esos pacientes.

Los valores ausentes de SFTPB se sustituyen por esa mediana tanto en train como en el fold de test.

El conjunto de test nunca participa en el cálculo de la mediana.

Esto evita data leakage.

---

## 8. Estandarización

Las variables numéricas tienen escalas diferentes.

Por ejemplo:

- edad en años
- tabaquismo en pack-years
- FVC y DLCO en porcentaje
- proteínas en NPX

Se aplica:

`z = (x - media_train) / desviacion_train`

mediante:

```python
StandardScaler()
```

La media y desviación estándar también se calculan únicamente con el training fold.

Esto permite que los coeficientes del modelo sean comparables y evita que una variable tenga mayor influencia únicamente por su escala numérica.

---

## 9. Variables categóricas

Las variables categóricas son:

- sexo
- patrón TC

Primero se imputan valores ausentes mediante la categoría más frecuente del training fold.

Después se aplica:

```python
OneHotEncoder(drop="first", handle_unknown="ignore")
```

Esto crea variables binarias para cada categoría.

No se asignan números ordinales arbitrarios a los patrones de TC.

---

## 10. Modelo predictivo

El modelo es:

```python
LogisticRegression(
    C=1.0,
    solver="liblinear",
    max_iter=5000
)
```

Se utiliza regresión logística con regularización L2.

La regresión calcula una puntuación:

`S = β0 + β1x1 + β2x2 + ... + βnxn`

y posteriormente la convierte en probabilidad:

`P = 1 / (1 + exp(-S))`

Los coeficientes `β` no se eligen manualmente.

Se estiman durante el entrenamiento buscando los valores que explican mejor la diferencia entre progresores y no progresores.

La regularización L2 penaliza coeficientes excesivamente grandes y ayuda a reducir overfitting.

---

## 11. Validación cruzada

Se utiliza:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=2026
)
```

La cohorte se divide en cinco grupos.

En cada iteración:

1. cuatro folds se utilizan para entrenamiento
2. un fold se utiliza para evaluación
3. se repite el proceso hasta que todos los pacientes han sido utilizados como test una vez

La estratificación mantiene una proporción similar de progresores y no progresores en cada fold.

---

## 12. Predicciones out-of-fold

Se utiliza:

```python
cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict_proba"
)
```

Cada paciente recibe una predicción generada por un modelo que no ha sido entrenado con ese paciente.

Al finalizar existen 650 probabilidades out-of-fold.

Estas probabilidades son las utilizadas para calcular las métricas de validación.

No se evalúa el rendimiento utilizando predicciones del mismo modelo sobre los pacientes con los que fue entrenado.

---

## 13. ROC-AUC

A partir de las 650 probabilidades out-of-fold se construye la curva ROC.

Para cada posible umbral de clasificación se calcula:

`Sensibilidad = TP / (TP + FN)`

`Especificidad = TN / (TN + FP)`

La curva ROC representa:

`Sensibilidad` frente a `1 - Especificidad`

El área bajo esta curva es el ROC-AUC.

Resultado del modelo final:

**ROC-AUC = 0,9558**

Interpretación:

si se selecciona al azar un progresor y un no progresor, el modelo asigna un riesgo superior al progresor aproximadamente en el 95,6% de las parejas.

---

## 14. Average Precision

También se calcula Average Precision.

Esta métrica resume la relación entre:

- precision
- recall

Resultado:

**Average Precision = 0,9378**

Sirve como métrica complementaria a ROC-AUC.

---

## 15. Brier Score

El Brier Score mide el error de las probabilidades predichas.

Se calcula como:

`Brier = media((p - y)^2)`

donde:

- `p` es la probabilidad predicha
- `y` es el resultado real, 0 o 1

Resultado:

**Brier Score = 0,0811**

Un valor menor indica probabilidades más próximas a los resultados reales.

---

## 16. Comparación de modelos

Se compararon tres configuraciones utilizando exactamente los mismos folds:

1. variables clínicas
2. variables clínicas + SFTPB + GDF15
3. variables clínicas + SFTPB + GDF15 + MMP7 + SFTPD

El objetivo no era únicamente maximizar el AUC, sino comprobar si la proteómica añadía información sobre un modelo clínico de base.

El modelo de cuatro proteínas se mantuvo como modelo final porque combina un rendimiento elevado con una firma molecular todavía pequeña e interpretable.

---

## 17. Definición de los grupos de riesgo

Los grupos se definen usando exclusivamente las probabilidades out-of-fold.

### Bajo riesgo

Se buscan puntos de la curva ROC con:

`Sensibilidad >= 95%`

Entre ellos se escoge el de mayor especificidad.

Resultado:

**cutoff = 0,279**

### Alto riesgo

Se buscan puntos con:

`Especificidad >= 90%`

Entre ellos se escoge el de mayor sensibilidad.

Resultado:

**cutoff = 0,481**

Por tanto:

- bajo riesgo: `p < 0,279`
- intermedio: `0,279 <= p < 0,481`
- alto riesgo: `p >= 0,481`

---

## 18. Tasas observadas de progresión

Una vez definidos los grupos se calcula:

`progresión observada = progresores / pacientes del grupo`

Resultados:

- bajo riesgo: **4,8%**
- riesgo intermedio: **39,0%**
- alto riesgo: **88,6%**

Estas cifras describen la cohorte de desarrollo y no deben interpretarse como tasas universales aplicables a cualquier población.

---

## 19. Separación entre entrenamiento y evaluación

Un punto crítico del análisis es que el preprocesado se encuentra dentro de un `Pipeline`.

Esto significa que en cada fold:

1. la imputación aprende solo del train
2. la estandarización aprende solo del train
3. el one-hot encoding se ajusta solo con train
4. el modelo se entrena solo con train
5. la evaluación se realiza sobre pacientes no vistos

Este diseño reduce el riesgo de data leakage.

---

## 20. Modelo final para la calculadora

Después de finalizar la evaluación interna, se ajusta una versión final del modelo con los 650 pacientes.

Ese ajuste se utiliza únicamente para:

- obtener coeficientes finales
- construir el prototipo de calculadora
- generar probabilidades para ejemplos individuales

El rendimiento reportado del modelo sigue siendo el obtenido mediante predicciones out-of-fold.

No se reporta el rendimiento del modelo evaluado sobre los mismos pacientes utilizados para entrenarlo.

---

## 21. Interpretación clínica

El modelo integra tres tipos de información:

### Estado del paciente

- edad
- sexo
- IMC
- tabaquismo

### Función y estructura pulmonar

- FVC
- DLCO
- patrón TC

### Biología molecular

- SFTPB
- GDF15
- MMP7
- SFTPD

La hipótesis es que estas fuentes contienen información complementaria sobre el riesgo de progresión.

---

## 22. Limitaciones

### Validación interna

Todo el análisis se realiza sobre una única cohorte.

La validación cruzada reduce el optimismo del rendimiento, pero no sustituye una cohorte externa.

### Selección de biomarcadores

La selección de la firma se ha realizado utilizando la misma cohorte de desarrollo.

Por ello, el AUC debe interpretarse como una estimación interna de rendimiento.

### Umbrales

Los puntos de corte 0,279 y 0,481 se han optimizado en esta cohorte.

Deben considerarse umbrales de desarrollo.

### Efectos técnicos

El dataset disponible no contiene identificadores de placa/lote suficientes para evaluar formalmente posibles efectos técnicos residuales de Olink.

---

## 23. Siguiente paso necesario

Antes de uso clínico sería necesario:

1. congelar variables, proteínas y coeficientes
2. aplicar el modelo sin modificarlo a una cohorte independiente
3. evaluar discriminación y calibración
4. comprobar estabilidad de los umbrales
5. realizar validación prospectiva multicéntrica

---

## 24. Reproducibilidad del repositorio

El código se separa en dos partes:

### `modelo_riesgo.py`

Contiene la definición del pipeline y del modelo.

### `calculos_resultados.py`

Reproduce:

- comparación de modelos
- predicciones out-of-fold
- ROC-AUC
- Average Precision
- Brier Score
- selección de umbrales
- grupos de riesgo
- CSV finales

Esta separación permite distinguir claramente el modelo predictivo de los cálculos utilizados para evaluar y presentar sus resultados.
