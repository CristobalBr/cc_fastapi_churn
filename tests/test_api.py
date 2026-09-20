from fastapi.testclient import TestClient

from app.main import app


VALID_OBSERVATION = {
    "tenure": 12,
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"


def test_model_info():
    with TestClient(app) as client:
        response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_type"] == "LogisticRegression"
    assert data["model_version"] == "1.0.0"
    assert data["python_version"] == "3.11.16"
    assert data["sklearn_version"] == "1.5.2"
    assert data["target"] == "Churn"
    assert data["positive_class"] == "Yes"
    assert len(data["features"]) == 19
    assert "accuracy" in data["metrics"]
    assert "roc_auc" in data["metrics"]


def test_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=VALID_OBSERVATION,
        )

    assert response.status_code == 200

    data = response.json()

    assert data["prediccion"] in ["Yes", "No"]
    assert 0 <= data["probabilidad"] <= 1
    assert data["model_version"] == "1.0.0"
    assert "timestamp" in data


def test_predict_batch():
    observations = [
        VALID_OBSERVATION,
        {
            **VALID_OBSERVATION,
            "tenure": 60,
            "MonthlyCharges": 45.0,
            "TotalCharges": 2700.0,
            "gender": "Female",
            "Partner": "Yes",
            "Dependents": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
        },
    ]

    with TestClient(app) as client:
        response = client.post(
            "/predict-batch",
            json=observations,
        )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2
    assert data[0]["prediccion"] in ["Yes", "No"]
    assert data[1]["prediccion"] in ["Yes", "No"]


def test_predict_invalid_input():
    invalid_observation = {
        **VALID_OBSERVATION,
        "tenure": -1,
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=invalid_observation,
        )

    assert response.status_code == 422


def test_predict_batch_invalid_input():
    invalid_observation = {
        **VALID_OBSERVATION,
        "MonthlyCharges": -10,
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict-batch",
            json=[VALID_OBSERVATION, invalid_observation],
        )

    assert response.status_code == 422
