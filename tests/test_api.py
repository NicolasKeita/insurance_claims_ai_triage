import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.anomaly_training import AnomalyAssessment
from ml.inference import RegisteredTriagePredictor
from test_ml_inference import FakePipeline, make_features


def fake_predictor_loader():
    return RegisteredTriagePredictor(FakePipeline(), "test-model", "1", "candidate")


class FakeAnomalyPredictor:
    registered_model = "test-anomaly-model"
    version = "2"
    alias = "candidate"

    def assess(self, features):
        assert features == make_features()
        return RegisteredAnomalyAssessment(
            assessment=AnomalyAssessment(
                is_anomalous=True,
                anomaly_score=0.72,
                threshold=0.62,
                reference_percentile=0.995,
                statistical_signals=["repair_amount: very_high"],
            ),
            registered_model=self.registered_model,
            model_version=self.version,
            model_alias=self.alias,
        )


def fake_anomaly_predictor_loader():
    return FakeAnomalyPredictor()


app = create_app(
    predictor_loader=fake_predictor_loader,
    anomaly_predictor_loader=fake_anomaly_predictor_loader,
)


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "models": {
                "triage": {
                    "name": "test-model",
                    "version": "1",
                    "alias": "candidate",
                },
                "anomaly": {
                    "name": "test-anomaly-model",
                    "version": "2",
                    "alias": "candidate",
                },
            },
        }


def test_predict_triage():
    with TestClient(app) as client:
        response = client.post(
            "/v1/triage/predict", json=make_features().model_dump(mode="json")
        )
        assert response.status_code == 200
        body = response.json()
        assert body["recommended_workflow"] == "STANDARD"
        assert body["confidence"] == 0.70


def test_assess_anomaly():
    with TestClient(app) as client:
        response = client.post(
            "/v1/anomaly/assess", json=make_features().model_dump(mode="json")
        )
        assert response.status_code == 200
        body = response.json()
        assert body["registered_model"] == "test-anomaly-model"
        assert body["model_version"] == "2"
        assert body["model_alias"] == "candidate"
        assert body["assessment"] == {
            "is_anomalous": True,
            "anomaly_score": 0.72,
            "threshold": 0.62,
            "reference_percentile": 0.995,
            "statistical_signals": ["repair_amount: very_high"],
        }


def test_combined_assessment_keeps_results_independent():
    with TestClient(app) as client:
        response = client.post(
            "/v1/claims/assess", json=make_features().model_dump(mode="json")
        )
        assert response.status_code == 200
        body = response.json()
        assert body["triage"]["recommended_workflow"] == "STANDARD"
        assert body["anomaly"]["assessment"]["is_anomalous"] is True
        assert body["anomaly"]["registered_model"] == "test-anomaly-model"


@pytest.mark.parametrize(
    "path",
    ["/v1/triage/predict", "/v1/anomaly/assess", "/v1/claims/assess"],
)
def test_invalid_features_are_rejected(path):
    with TestClient(app) as client:
        response = client.post(path, json={"vehicle_age": -5, "repair_amount": -100})
        assert response.status_code == 422


def test_models_are_loaded_once_per_lifespan():
    load_counts = {"triage": 0, "anomaly": 0}

    def load_triage():
        load_counts["triage"] += 1
        return fake_predictor_loader()

    def load_anomaly():
        load_counts["anomaly"] += 1
        return fake_anomaly_predictor_loader()

    counted_app = create_app(load_triage, load_anomaly)
    with TestClient(counted_app) as client:
        client.get("/health")
        client.post("/v1/claims/assess", json=make_features().model_dump(mode="json"))
        client.post("/v1/anomaly/assess", json=make_features().model_dump(mode="json"))

    assert load_counts == {"triage": 1, "anomaly": 1}
