"""Test errors."""

import pytest
from fastapi.testclient import TestClient

from agent.service import AgentRunError
from api.main import AppConfig, create_app
from tests.agent.api._support import FakeRunStore, fake_claim_loader, make_run


@pytest.mark.parametrize("failure_kind,expected_status", [("GROUNDING", 502), ("GENERATION", 503), ("TOOL", 503)])
def test_failed_run_id_and_sanitized_http_error(failure_kind, expected_status, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    store = FakeRunStore()

    class BrokenAgent:
        def review_claim(self, claim_id, objective=None):
            run = make_run(claim_id, objective, failed=True)
            store.save(run)
            raise AgentRunError("secret raw LLM output", run_id=run.run_id, failure_kind=failure_kind)

    app = create_app(
        AppConfig(
            agent_service_loader=lambda *args, **kwargs: BrokenAgent(),
            agent_claim_loader=fake_claim_loader,
            agent_run_store_loader=lambda _: store,
        )
    )
    with TestClient(app) as client:
        response = client.post("/v1/claims/demo/agent/review")
        assert response.status_code == expected_status
        assert "secret" not in response.text
        run_id = response.json()["detail"]["run_id"]
        stored = client.get(f"/v1/agent/runs/{run_id}")
        assert stored.status_code == 200 and stored.json()["status"] == "FAILED"


def test_audit_reads_do_not_load_agent_and_bounded_limit(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    store = FakeRunStore()
    store.save(make_run())

    def forbidden(*args, **kwargs):
        raise AssertionError("Audit reads must not load a model")

    app = create_app(
        AppConfig(
            agent_service_loader=forbidden, agent_claim_loader=fake_claim_loader, agent_run_store_loader=lambda _: store
        )
    )
    with TestClient(app) as client:
        assert client.get("/v1/claims/demo/agent/runs").status_code == 200
        assert client.get("/v1/claims/missing/agent/runs").status_code == 404
        for limit in ("0", "101", "1.5", "true"):
            assert client.get(f"/v1/claims/demo/agent/runs?limit={limit}").status_code == 422


def test_database_unconfigured_and_loader_failure_are_controlled(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with TestClient(create_app()) as client:
        for method, path in (
            ("post", "/v1/claims/demo/agent/review"),
            ("get", "/v1/claims/demo/agent/runs"),
            ("get", "/v1/agent/runs/demo"),
        ):
            response = getattr(client, method)(path)
            assert response.status_code == 503 and response.json()["detail"] == "Database is not configured"

    def broken(*args, **kwargs):
        raise RuntimeError("secret connection details")

    with TestClient(create_app(AppConfig(agent_service_loader=broken, agent_claim_loader=fake_claim_loader))) as client:
        response = client.post("/v1/claims/demo/agent/review")
        assert response.status_code == 503 and "secret" not in response.text
