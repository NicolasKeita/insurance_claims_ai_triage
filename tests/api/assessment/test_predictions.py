"""Test predictions."""

from fastapi.testclient import TestClient

from claims.investigation.policy import DEFAULT_INVESTIGATION_POLICY
from tests.api.assessment._constants import app
from tests.api.assessment._support import make_combined_payload
from tests.intelligence.ml.triage._inference_fakes import make_features


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
            "investigation_policy_version": DEFAULT_INVESTIGATION_POLICY.version,
        }


def test_predict_triage():
    with TestClient(app) as client:
        response = client.post("/v1/triage/predict", json=make_features().model_dump(mode="json"))
        assert response.status_code == 200
        body = response.json()
        assert body["recommended_workflow"] == "STANDARD"
        assert body["confidence"] == 0.70


def test_assess_anomaly():
    with TestClient(app) as client:
        response = client.post("/v1/anomaly/assess", json=make_features().model_dump(mode="json"))
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
        response = client.post("/v1/claims/assess", json=make_combined_payload())
        assert response.status_code == 200
        body = response.json()
        assert body["triage"]["recommended_workflow"] == "STANDARD"
        assert body["anomaly"]["assessment"]["is_anomalous"] is True
        assert body["anomaly"]["registered_model"] == "test-anomaly-model"
        assert body["investigation"]["policy_version"] == DEFAULT_INVESTIGATION_POLICY.version
        codes = {signal["code"] for signal in body["investigation"]["signals"]}
        assert "DOCUMENT_FIELD_MISMATCH" in codes
        assert "ADDITIONAL_QUOTE_DAMAGE" in codes
        assert "UNMAPPED_QUOTE_ITEM" in codes
        assert "ANOMALOUS_PATTERN" in codes
        mismatch = next(
            signal for signal in body["investigation"]["signals"] if signal["code"] == "DOCUMENT_FIELD_MISMATCH"
        )
        assert mismatch["evidence"]["field"] == "incident_date"
        assert mismatch["evidence"]["matched"] is False
