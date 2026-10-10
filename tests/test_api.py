import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from claims.consistency import ClaimConsistencyReport
from investigation.policy import DEFAULT_INVESTIGATION_POLICY
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
        assert features in (make_features(), make_combined_features())
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


def make_consistency():
    return ClaimConsistencyReport(
        claim_id_match=True,
        incident_date_match=False,
        location_match=True,
        vehicle_match=True,
        collision_type_match=True,
        injuries_match=True,
        declared_damage_match=True,
        additional_quote_damage=("HOOD",),
        unmapped_quote_items=("paint material",),
    )


def make_combined_features():
    return make_features().model_copy(update={"consistency_issue_count": 3})


def make_combined_payload():
    return {
        "features": make_combined_features().model_dump(mode="json"),
        "consistency": make_consistency().model_dump(mode="json"),
    }


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
            "investigation_policy_version": DEFAULT_INVESTIGATION_POLICY.version,
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
            "/v1/claims/assess", json=make_combined_payload()
        )
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
            signal
            for signal in body["investigation"]["signals"]
            if signal["code"] == "DOCUMENT_FIELD_MISMATCH"
        )
        assert mismatch["evidence"]["field"] == "incident_date"
        assert mismatch["evidence"]["matched"] is False


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
        response = client.post(
            "/v1/claims/assess", json=make_features().model_dump(mode="json")
        )
        assert response.status_code == 422


def test_combined_assessment_rejects_conflicting_consistency_count():
    with TestClient(app) as client:
        payload = make_combined_payload()
        payload["features"]["consistency_issue_count"] = 0
        response = client.post("/v1/claims/assess", json=payload)
        assert response.status_code == 422


@pytest.mark.parametrize(
    "path, payload",
    [
        ("/v1/triage/predict", {"vehicle_age": -5, "repair_amount": -100}),
        ("/v1/anomaly/assess", {"vehicle_age": -5, "repair_amount": -100}),
        (
            "/v1/investigation/assess",
            {
                "features": {"vehicle_age": -5, "repair_amount": -100},
                "consistency": make_consistency().model_dump(mode="json"),
                "anomaly": {
                    "is_anomalous": False,
                    "anomaly_score": 0.4,
                    "threshold": 0.6,
                    "reference_percentile": 0.5,
                },
            },
        ),
        (
            "/v1/claims/assess",
            {
                "features": {"vehicle_age": -5, "repair_amount": -100},
                "consistency": make_consistency().model_dump(mode="json"),
            },
        ),
    ],
)
def test_invalid_features_are_rejected(path, payload):
    with TestClient(app) as client:
        response = client.post(path, json=payload)
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
        client.post("/v1/claims/assess", json=make_combined_payload())
        client.post("/v1/anomaly/assess", json=make_features().model_dump(mode="json"))

    assert load_counts == {"triage": 1, "anomaly": 1}


def test_history_database_unconfigured_is_503_and_loads_no_models(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def forbidden():
        raise AssertionError("Read-only history must not load models")

    with TestClient(create_app(forbidden, forbidden)) as client:
        response = client.get("/v1/claims/any/history")
        assert response.status_code == 503
        assert response.json() == {"detail": "Database is not configured"}


def test_history_database_failure_is_503_not_empty_profile():
    from contextlib import contextmanager
    from sqlalchemy.exc import OperationalError

    class BrokenFactory:
        @contextmanager
        def begin(self):
            raise OperationalError("connect", {}, Exception("unavailable"))
            yield

    with TestClient(create_app(session_factory=BrokenFactory())) as client:
        response = client.get("/v1/claims/any/history")
        assert response.status_code == 503
        assert response.json() == {"detail": "Database operation failed"}
