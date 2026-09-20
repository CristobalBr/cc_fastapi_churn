from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Observacion(BaseModel):
    tenure: int = Field(..., ge=0, le=72)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service",
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No",
    ]

    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_loaded: bool
    model_version: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    prediccion: str
    probabilidad: float = Field(..., ge=0, le=1)
    model_version: str
    timestamp: datetime


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_type: str
    model_version: str
    python_version: str
    sklearn_version: str
    target: str
    positive_class: str
    features: list[str]
    metrics: dict[str, float]