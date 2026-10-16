"""Read-only stored vision HTTP route; no inference path is available."""

from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError
from test_multimodal_investigation import make_visual_assessment

from api.main import create_app
from persistence.repositories import ClaimNotFoundError
from persistence.vision_assessments import StoredVisionAssessment


def test_stored_vision_read_loads_once_without_any_predictor():
    loaded = []
    record = StoredVisionAssessment(uuid4(), datetime.now(timezone.utc), make_visual_assessment())

    class Store:
        def latest_for_claim(self, claim_id):
            if claim_id == "missing":
                raise ClaimNotFoundError("Claim missing does not exist")
            return None if claim_id == "unassessed" else record

    def forbidden():
        raise AssertionError("A stored visual read must never load a predictor")

    def loader(factory):
        loaded.append(factory)
        return Store()

    with TestClient(
        create_app(predictor_loader=forbidden, anomaly_predictor_loader=forbidden, vision_store_loader=loader)
    ) as client:
        for _ in range(2):
            response = client.get("/v1/claims/CLAIM-2026-00001/vision/latest")
            assert response.status_code == 200
            assert response.json()["result"]["fusion_policy_version"] == "vision_fusion_v1"
            assert response.json()["result"]["consistency"]["additional_visual_damage"] == ["HOOD"]
        assert client.get("/v1/claims/unassessed/vision/latest").status_code == 404
        assert client.get("/v1/claims/missing/vision/latest").status_code == 404
    assert len(loaded) == 1


def test_stored_vision_api_requires_database_without_loading_models(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with TestClient(create_app()) as client:
        assert client.get("/v1/claims/CLAIM-2026-00001/vision/latest").status_code == 503


def test_stored_vision_database_failure_is_sanitized_and_never_returns_an_assessment():
    class UnavailableStore:
        def latest_for_claim(self, claim_id):
            raise SQLAlchemyError("private database connection detail")

    with TestClient(create_app(vision_store_loader=lambda _: UnavailableStore())) as client:
        response = client.get("/v1/claims/CLAIM-2026-00001/vision/latest")
    assert response.status_code == 503
    assert response.json() == {"detail": "Database operation failed"}
    assert "private" not in response.text
