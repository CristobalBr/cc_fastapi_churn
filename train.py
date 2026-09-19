from pathlib import Path
import json
import sys

import joblib
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"


# Configuración
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Churn"


# Variables utilizadas
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# Carga de datos
df = pd.read_csv(DATA_PATH)


# Conversión de TotalCharges a numérico
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# Variables predictoras y objetivo
X = df[FEATURES].copy()
y = df[TARGET].copy()


# Separación entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


# Preprocesamiento numérico
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)


# Preprocesamiento categórico
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)


# Unión de ambos tipos de variables
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ]
)


# Pipeline completo
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# Entrenamiento
pipeline.fit(X_train, y_train)


# Predicciones
y_pred = pipeline.predict(X_test)

positive_class_index = list(
    pipeline.named_steps["model"].classes_
).index("Yes")

y_proba = pipeline.predict_proba(X_test)[:, positive_class_index]


# Métricas
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(
        y_test,
        y_pred,
        pos_label="Yes",
        zero_division=0,
    ),
    "recall": recall_score(
        y_test,
        y_pred,
        pos_label="Yes",
        zero_division=0,
    ),
    "f1": f1_score(
        y_test,
        y_pred,
        pos_label="Yes",
        zero_division=0,
    ),
    "roc_auc": roc_auc_score(
        (y_test == "Yes").astype(int),
        y_proba,
    ),
}


# Guardar modelo
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(pipeline, MODEL_PATH)


# Metadatos
metadata = {
    "model_type": "LogisticRegression",
    "model_version": "1.0.0",
    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    "target": TARGET,
    "positive_class": "Yes",
    "sklearn_version": sklearn.__version__,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "features": FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "metrics": {
        name: round(value, 4)
        for name, value in metrics.items()
    },
}


with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2, ensure_ascii=False)


# Resumen
print("Entrenamiento finalizado")
print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

print("\nMétricas:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print(f"\nModelo guardado en: {MODEL_PATH}")
print(f"Metadata guardada en: {METADATA_PATH}")