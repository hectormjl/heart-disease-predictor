"""Tests for the /predict endpoint: valid inputs and validation errors.

Uses FastAPI's TestClient, which calls the app directly in-process (no real
network socket, no need to run `uvicorn` separately). Wrapping it in a `with`
block matters here specifically: it triggers the app's `lifespan` handler, so
the real trained model gets loaded from models/model.pkl before any request
is made - without it, MODEL_STATE would stay empty and every request would
hit the 503 "model not loaded" branch in api/main.py.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

VALID_PATIENT = {
    "age": 63,
    "sex": "male",
    "cp": "typical angina",
    "trestbps": 145,
    "chol": 233,
    "fbs": True,
    "restecg": "lv hypertrophy",
    "thalch": 150,
    "exang": False,
    "oldpeak": 2.3,
    "slope": "downsloping",
    "ca": 0,
    "thal": "fixed defect",
}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_predict_valid_patient_returns_200(client):
    response = client.post("/predict", json=VALID_PATIENT)

    assert response.status_code == 200


def test_predict_valid_patient_returns_expected_shape(client):
    response = client.post("/predict", json=VALID_PATIENT)
    body = response.json()

    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["risk_class"] in ("low_risk", "high_risk")
    assert "disclaimer" in body


def test_predict_without_optional_fields_still_works(client):
    # slope/ca/thal are optional in PatientInput: the pipeline is trained to
    # impute them when missing, so dropping them should not break a request.
    patient = {k: v for k, v in VALID_PATIENT.items() if k not in ("slope", "ca", "thal")}

    response = client.post("/predict", json=patient)

    assert response.status_code == 200


def test_predict_missing_required_field_returns_422(client):
    patient = {k: v for k, v in VALID_PATIENT.items() if k != "age"}

    response = client.post("/predict", json=patient)

    assert response.status_code == 422


def test_predict_age_out_of_range_returns_422(client):
    patient = {**VALID_PATIENT, "age": 200}

    response = client.post("/predict", json=patient)

    assert response.status_code == 422


def test_predict_unknown_chest_pain_type_returns_422(client):
    patient = {**VALID_PATIENT, "cp": "not_a_real_type"}

    response = client.post("/predict", json=patient)

    assert response.status_code == 422


def test_predict_wrong_field_type_returns_422(client):
    patient = {**VALID_PATIENT, "trestbps": "high"}

    response = client.post("/predict", json=patient)

    assert response.status_code == 422
