"""Test lifespan."""

import pytest
from fastapi.testclient import TestClient

from api.main import AppConfig, create_app
from tests.agent.api._support import (
    fake_claim_loader,
    lifespan_loaders,
    retrieval_doubles,
)


@pytest.mark.parametrize("preload", [True, False])
def test_agent_reuses_lifespan_retrieval_caches_and_cleanup(preload, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    loaded, passed = [], []

    Similar, Knowledge = retrieval_doubles()
    similar, knowledge = Similar(), Knowledge()

    similar_loader, knowledge_loader, agent_loader = lifespan_loaders(loaded, passed, similar, knowledge)
    app = create_app(
        AppConfig(
            agent_service_loader=agent_loader,
            agent_claim_loader=fake_claim_loader,
            similar_service_loader=similar_loader,
            knowledge_retriever_loader=knowledge_loader,
        )
    )
    with TestClient(app) as client:

        def retrieval_requests():
            assert client.get("/v1/claims/demo/similar").status_code == 200
            assert (
                client.post("/v1/knowledge/search", json={"query": "test", "source_type": "POLICY"}).status_code == 200
            )

        if preload:
            retrieval_requests()
        for _ in range(2):
            assert client.post("/v1/claims/demo/agent/review").status_code == 200
        if not preload:
            retrieval_requests()
    assert loaded == ["similar", "knowledge"]
    assert len(passed) == 1
    assert passed[0]["similar_service"] is (similar if preload else None)
    assert passed[0]["knowledge_retriever"] is (knowledge if preload else None)
    assert similar.closed and knowledge.closed
