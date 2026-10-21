"""Test lifecycle."""

from fastapi.testclient import TestClient

from api.main import AppConfig, create_app
from tests.api.assessment._support import (
    fake_anomaly_predictor_loader,
    fake_predictor_loader,
    make_combined_payload,
)
from tests.intelligence.ml.triage._inference_fakes import make_features


def test_models_are_loaded_once_per_lifespan():
    load_counts = {"triage": 0, "anomaly": 0}

    def load_triage():
        load_counts["triage"] += 1
        return fake_predictor_loader()

    def load_anomaly():
        load_counts["anomaly"] += 1
        return fake_anomaly_predictor_loader()

    counted_app = create_app(AppConfig(predictor_loader=load_triage, anomaly_predictor_loader=load_anomaly))
    with TestClient(counted_app) as client:
        client.get("/health")
        client.post("/v1/claims/assess", json=make_combined_payload())
        client.post("/v1/anomaly/assess", json=make_features().model_dump(mode="json"))

    assert load_counts == {"triage": 1, "anomaly": 1}


def test_history_database_unconfigured_is_503_and_loads_no_models(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def forbidden():
        raise AssertionError("Read-only history must not load models")

    with TestClient(create_app(AppConfig(predictor_loader=forbidden, anomaly_predictor_loader=forbidden))) as client:
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

    with TestClient(create_app(AppConfig(session_factory=BrokenFactory()))) as client:
        response = client.get("/v1/claims/any/history")
        assert response.status_code == 503
        assert response.json() == {"detail": "Database operation failed"}
