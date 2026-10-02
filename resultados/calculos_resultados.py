import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
)

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

DATA = "data/reto_fibrosis_pulmonar_dataset_participantes.tsv"
OUT_DIR = "resultados"

# Semilla fija para que los folds sean reproducibles
RANDOM_STATE = 2026

# ------------------------------------------------------------
# CARGA DE DATOS
# ------------------------------------------------------------

df = pd.read_csv(DATA, sep="\t")

# Cohorte principal:
# solo pacientes con ILA en el momento basal
cohort = df[df["clinic_group"].isin([
    "Non_stable_fibrotic_ILA",
    "Stable_fibrotic_ILA",
    "Progressive_ILA"
])].copy()

# Variable objetivo:
# 1 = progresor
# 0 = no progresor
cohort["target"] = (
    cohort["clinic_group"] == "Progressive_ILA"
).astype(int)

y = cohort["target"].to_numpy()

# ------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------

clinical_numeric = [
    "age",
    "cigarette_packs_year",
    "bmi",
    "fvc_basal",
    "dlco_basal",
]

clinical_categorical = [
    "sex",
    "ct_pattern",
]

proteins_2 = [
    "NPX_SFTPB",
    "NPX_GDF15",
]

proteins_4 = [
    "NPX_SFTPB",
    "NPX_GDF15",
    "NPX_MMP7",
    "NPX_SFTPD",
]

# ------------------------------------------------------------
# FUNCIÓN PARA CREAR EL PIPELINE
# ------------------------------------------------------------

def build_pipeline(numeric_columns):
    """
    Crea el pipeline completo:
    - imputación por mediana para variables numéricas
    - estandarización
    - imputación por moda para categóricas
    - one-hot encoding
    - regresión logística regularizada L2
    """

    preprocess = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_columns,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore",
                    ),
                ),
            ]),
            clinical_categorical,
        ),
    ])

    return Pipeline([
        ("preprocess", preprocess),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=5000,
            ),
        ),
    ])


# ------------------------------------------------------------
# VALIDACIÓN CRUZADA
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# ------------------------------------------------------------
# 1. COMPARACIÓN DE MODELOS
# ------------------------------------------------------------

models = {
    "Clinical": clinical_numeric,
    "Clinical + 2 proteins": clinical_numeric + proteins_2,
    "Clinical + 4 proteins": clinical_numeric + proteins_4,
}

model_results = []
predictions = {}

for model_name, numeric_columns in models.items():

    X = cohort[numeric_columns + clinical_categorical]

    model = build_pipeline(numeric_columns)

    # Predicción out-of-fold:
    # cada paciente es predicho por un modelo
    # que NO ha sido entrenado con él
    p_oof = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    predictions[model_name] = p_oof

    model_results.append({
        "model": model_name,
        "OOF_AUC": roc_auc_score(y, p_oof),
        "Average_precision": average_precision_score(y, p_oof),
        "Brier_score": brier_score_loss(y, p_oof),
    })

comparison_df = pd.DataFrame(model_results)

# ------------------------------------------------------------
# 2. MÉTRICAS DEL MODELO FINAL
# ------------------------------------------------------------

final_predictions = predictions["Clinical + 4 proteins"]

auc = roc_auc_score(y, final_predictions)
average_precision = average_precision_score(y, final_predictions)
brier = brier_score_loss(y, final_predictions)

# Curva ROC para obtener sensibilidad/especificidad
fpr, tpr, thresholds = roc_curve(y, final_predictions)

specificity = 1 - fpr

# ------------------------------------------------------------
# UMBRAL DE BAJO RIESGO
# ------------------------------------------------------------
# Buscamos sensibilidad >= 95%.
# De esos puntos elegimos el de mayor especificidad.

low_candidates = np.where(tpr >= 0.95)[0]

low_idx = low_candidates[
    np.argmin(fpr[low_candidates])
]

low_cutoff = thresholds[low_idx]

low_sensitivity = tpr[low_idx]
low_specificity = specificity[low_idx]

# ------------------------------------------------------------
# UMBRAL DE ALTO RIESGO
# ------------------------------------------------------------
# Buscamos especificidad >= 90%.
# De esos puntos elegimos el de mayor sensibilidad.

high_candidates = np.where(specificity >= 0.90)[0]

high_idx = high_candidates[
    np.argmax(tpr[high_candidates])
]

high_cutoff = thresholds[high_idx]

high_sensitivity = tpr[high_idx]
high_specificity = specificity[high_idx]

metrics_df = pd.DataFrame([{
    "ROC_AUC": auc,
    "Average_precision": average_precision,
    "Brier_score": brier,

    "low_cutoff": low_cutoff,
    "low_sensitivity": low_sensitivity,
    "low_specificity": low_specificity,

    "high_cutoff": high_cutoff,
    "high_sensitivity": high_sensitivity,
    "high_specificity": high_specificity,
}])

# ------------------------------------------------------------
# 3. GRUPOS DE RIESGO
# ------------------------------------------------------------

risk_group = np.where(
    final_predictions < low_cutoff,
    "Low",
    np.where(
        final_predictions >= high_cutoff,
        "High",
        "Intermediate",
    ),
)

risk_rows = []

for group in ["Low", "Intermediate", "High"]:

    mask = risk_group == group

    n = int(mask.sum())
    progressors = int(y[mask].sum())
    non_progressors = int(n - progressors)

    progression_rate = (
        progressors / n
        if n > 0
        else np.nan
    )

    risk_rows.append({
        "risk_group": group,
        "n": n,
        "progressors": progressors,
        "non_progressors": non_progressors,
        "progression_rate": progression_rate,
        "progression_rate_percent": progression_rate * 100,
    })

risk_df = pd.DataFrame(risk_rows)

# ------------------------------------------------------------
# 4. GUARDAR RESULTADOS
# ------------------------------------------------------------

from pathlib import Path

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

comparison_df.to_csv(
    f"{OUT_DIR}/comparacion_modelos.csv",
    index=False,
)

metrics_df.to_csv(
    f"{OUT_DIR}/metricas_modelo.csv",
    index=False,
)

risk_df.to_csv(
    f"{OUT_DIR}/grupos_riesgo.csv",
    index=False,
)

# ------------------------------------------------------------
# 5. MOSTRAR RESULTADOS EN TERMINAL
# ------------------------------------------------------------

print("\n=== COMPARACIÓN DE MODELOS ===")
print(comparison_df.to_string(index=False))

print("\n=== MÉTRICAS MODELO FINAL ===")
print(metrics_df.to_string(index=False))

print("\n=== GRUPOS DE RIESGO ===")
print(risk_df.to_string(index=False))
