import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

DATA = "data/reto_fibrosis_pulmonar_dataset_participantes.tsv"

df = pd.read_csv(DATA, sep="\\t")

cohort = df[df["clinic_group"].isin([
    "Non_stable_fibrotic_ILA",
    "Stable_fibrotic_ILA",
    "Progressive_ILA"
])].copy()

cohort["target"] = (cohort["clinic_group"] == "Progressive_ILA").astype(int)

# All baseline clinical variables retained, even if some add little predictive value
numeric = [
    "age",
    "cigarette_packs_year",
    "bmi",
    "fvc_basal",
    "dlco_basal",
    "NPX_SFTPB",
    "NPX_GDF15",
    "NPX_MMP7",
    "NPX_SFTPD",
]

categorical = [
    "sex",
    "ct_pattern"
]

X = cohort[numeric + categorical]
y = cohort["target"].to_numpy()

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ]), categorical),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=5000
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=2026
)

p_oof = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict_proba"
)[:, 1]

print("ROC-AUC:", roc_auc_score(y, p_oof))
print("Average precision:", average_precision_score(y, p_oof))
print("Brier score:", brier_score_loss(y, p_oof))

fpr, tpr, thresholds = roc_curve(y, p_oof)
specificity = 1 - fpr

valid_low = np.where(tpr >= 0.95)[0]
low_idx = valid_low[np.argmin(fpr[valid_low])]
low_cut = thresholds[low_idx]

valid_high = np.where(specificity >= 0.90)[0]
high_idx = valid_high[np.argmax(tpr[valid_high])]
high_cut = thresholds[high_idx]

risk_group = np.where(
    p_oof < low_cut,
    "Low",
    np.where(p_oof >= high_cut, "High", "Intermediate")
)

print("Low-risk cutoff:", low_cut)
print("High-risk cutoff:", high_cut)

for group in ["Low", "Intermediate", "High"]:
    mask = risk_group == group
    print(
        group,
        "n =", mask.sum(),
        "progression =", y[mask].mean()
    )


model.fit(X, y)
