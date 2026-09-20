from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    Observacion,
    PredictionResponse,
)


# Rutas
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
METADATA_PATH = PROJECT_ROOT / "model" / "metadata.json"


# Artefactos cargados en memoria
ARTIFACTS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    ARTIFACTS["model"] = model
    ARTIFACTS["metadata"] = metadata

    yield

    ARTIFACTS.clear()


app = FastAPI(
    title="Telco Customer Churn API",
    description="API para predecir abandono de clientes.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def root():
    return {
        "message": "Telco Customer Churn API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health"
    }

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    metadata = ARTIFACTS["metadata"]

    return {
        "status": "ok",
        "model_loaded": "model" in ARTIFACTS,
        "model_version": metadata["model_version"],
    }


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info():
    metadata = ARTIFACTS["metadata"]

    return {
        "model_type": metadata["model_type"],
        "model_version": metadata["model_version"],
        "python_version": metadata["python_version"],
        "sklearn_version": metadata["sklearn_version"],
        "target": metadata["target"],
        "positive_class": metadata["positive_class"],
        "features": metadata["features"],
        "metrics": metadata["metrics"],
    }


def generar_prediccion(obs: Observacion):
    model = ARTIFACTS["model"]
    metadata = ARTIFACTS["metadata"]

    df = pd.DataFrame([obs.model_dump()])

    prediccion = str(model.predict(df)[0])

    classes = list(
        model.named_steps["model"].classes_
    )

    positive_index = classes.index(
        metadata["positive_class"]
    )

    probabilidad = model.predict_proba(df)[0][positive_index]

    return {
        "prediccion": prediccion,
        "probabilidad": round(float(probabilidad), 4),
        "model_version": metadata["model_version"],
        "timestamp": datetime.now(timezone.utc),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(obs: Observacion):
    try:
        return generar_prediccion(obs)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Error al generar la predicción",
        ) from exc


@app.post(
    "/predict-batch",
    response_model=list[PredictionResponse],
)
def predict_batch(observaciones: list[Observacion]):
    try:
        return [
            generar_prediccion(obs)
            for obs in observaciones
        ]

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Error al generar las predicciones",
        ) from exc