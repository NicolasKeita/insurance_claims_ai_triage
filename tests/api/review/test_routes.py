"""HTTP review contract with controlled repository snapshots."""

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from api.main import AppConfig, create_app
from tests.api.review._support import configure_repositories, forbidden


def test_claim_found_partial_assessments_and_missing_claim(monkeypatch, reference_claim):
    service, _, reads = configure_repositories(monkeypatch, reference_claim)
    with TestClient(create_app(_config(service))) as client:
        response = client.get(f"/v1/claims/{reference_claim.claim_id}/review")
        assert response.status_code == 200
        view = response.json()
        assert view["claim"]["claim_id"] == reference_claim.claim_id
        assert view["triage"]["status"] == "NOT_EXECUTED"
        assert view["anomaly"]["data"] is None
        assert view["vision"]["status"] == "NOT_EXECUTED"
        assert view["vision"]["data"]["images"][0]["image_unavailable_reason"]
        assert view["history"]["status"] == "AVAILABLE"
        assert view["history"]["data"]["previous_claim_count"] == 0
        assert view["consistency"]["status"] == "UNAVAILABLE"
        assert client.get("/v1/claims/missing/review").status_code == 404
        assert client.get("/v1/review/demo-claims").json() == []
    assert reads == ["claim", "triage", "anomaly", "investigation", "history", "vision", "agent", "claim"]


def test_all_assessments_evidence_ids_and_model_metadata(monkeypatch, reference_claim):
    service, records, _ = configure_repositories(monkeypatch, reference_claim, complete=True)
    loaded = []
    config = _config(service)
    config.review_service_loader = lambda factory: loaded.append(factory) or service
    with TestClient(create_app(config)) as client:
        for _ in range(2):
            response = client.get(f"/v1/claims/{reference_claim.claim_id}/review")
            assert response.status_code == 200
        view = response.json()
    assert len(loaded) == 1
    assert view["triage"]["data"]["model_version"] == "triage-3"
    assert view["anomaly"]["data"]["assessment"]["is_anomalous"] is True
    assert view["audit"]["model_versions"]["damage"] == "damage-v1"
    assert view["audit"]["policy_versions"]["vision_fusion"] == "vision_fusion_v1"
    assert view["audit"]["timestamps"]["triage"]
    evidence_ids = {item["id"] for item in view["evidence"]}
    assert f"claim:{reference_claim.claim_id}" in evidence_ids
    assert any(identifier.startswith("vision:") and ":obs_1" in identifier for identifier in evidence_ids)
    assert "knowledge:demo-policy-v1-collision-0" in evidence_ids
    assert set(view["agent"]["data"]["recommendation"]["evidence_ids"]) <= evidence_ids
    assert view["knowledge"]["data"]["sources"][0]["source_type"] == "POLICY"
    assert view["knowledge"]["data"]["sources"][0]["similarity_score"] is None
    assert view["agent"]["data"]["recommendation"]["human_review_required"] is True
    assert "trace" not in view["agent"]["data"]
    assert records["agent"].trace


def test_database_unconfigured_and_failure_are_safe(monkeypatch, reference_claim):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("CLAIMS_REVIEW_DEMO", raising=False)
    with TestClient(create_app(AppConfig(predictor_loader=forbidden))) as client:
        assert client.get(f"/v1/claims/{reference_claim.claim_id}/review").status_code == 503

    class Service:
        def get_review(self, claim_id):
            raise SQLAlchemyError("private connection password")

    with TestClient(create_app(_config(Service()))) as client:
        response = client.get(f"/v1/claims/{reference_claim.claim_id}/review")
    assert response.status_code == 503
    assert "private" not in response.text


def _config(service):
    return AppConfig(
        review_service_loader=lambda _: service,
        predictor_loader=forbidden,
        anomaly_predictor_loader=forbidden,
        similar_service_loader=forbidden,
        knowledge_retriever_loader=forbidden,
        knowledge_rag_loader=forbidden,
        agent_service_loader=forbidden,
    )
