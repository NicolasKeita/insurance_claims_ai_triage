"""Test investigation."""

from fastapi.testclient import TestClient

from investigation.policy import DEFAULT_INVESTIGATION_POLICY
from ml.anomaly_training import AnomalyAssessment
from tests.api.assessment._constants import app
from tests.api.assessment._support import make_combined_payload
from tests.ml.triage._inference_fakes import make_features


def test_investigation_assessment_uses_structured_evidence():
    with TestClient(app) as client:
        payload = make_combined_payload()
        payload["anomaly"] = AnomalyAssessment(
            is_anomalous=True,
            anomaly_score=0.72,
            threshold=0.62,
            reference_percentile=0.995,
            statistical_signals=["repair_amount: very_high"],
        ).model_dump(mode="json")
        response = client.post("/v1/investigation/assess", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["policy_version"] == DEFAULT_INVESTIGATION_POLICY.version
        assert 0 <= body["review_score"] <= 100
        assert body["review_priority"] in {"NORMAL", "ELEVATED", "HIGH"}
        assert isinstance(body["review_recommended"], bool)
        assert all(signal["evidence"] for signal in body["signals"])


def test_combined_assessment_requires_consistency_details():
    with TestClient(app) as client:
        response = client.post("/v1/claims/assess", json=make_features().model_dump(mode="json"))
        assert response.status_code == 422


def test_combined_assessment_rejects_conflicting_consistency_count():
    with TestClient(app) as client:
        payload = make_combined_payload()
        payload["features"]["consistency_issue_count"] = 0
        response = client.post("/v1/claims/assess", json=payload)
        assert response.status_code == 422
