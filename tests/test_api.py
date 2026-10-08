from fastapi.testclient import TestClient

from api.main import create_app
from ml.inference import RegisteredTriagePredictor
from test_ml_inference import FakePipeline, make_features


def fake_predictor_loader():
    return RegisteredTriagePredictor(FakePipeline(), "test-model", "1", "candidate")


app = create_app(predictor_loader=fake_predictor_loader)


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok", "model": "test-model", "version": "1", "alias": "candidate"
        }


def test_predict_triage():
    with TestClient(app) as client:
        response = client.post("/v1/triage/predict", json=make_features().model_dump(mode="json"))
        assert response.status_code == 200
        body = response.json()
        assert body["recommended_workflow"] == "STANDARD"
        assert body["confidence"] == 0.70


def test_invalid_features_are_rejected():
    with TestClient(app) as client:
        response = client.post("/v1/triage/predict", json={"vehicle_age": -5, "repair_amount": -100})
        assert response.status_code == 422
