"""Test lifecycle."""

import pytest
from fastapi.testclient import TestClient

from api.main import AppConfig, create_app
from tests.intelligence.knowledge.api._support import FakeRag, FakeSearch


def test_search_answer_loaders_cached_and_lifespan_cleanup():
    search, rag = FakeSearch(), FakeRag()
    loaded = []

    def loader():
        loaded.append("retriever")
        return search

    def rag_loader(retriever):
        assert retriever is search
        loaded.append("rag")
        return rag

    with TestClient(
        create_app(AppConfig(knowledge_retriever_loader=loader, knowledge_rag_loader=rag_loader))
    ) as client:
        assert loaded == []
        for _ in range(2):
            response = client.post(
                "/v1/knowledge/search",
                json={"query": "test", "source_type": "POLICY", "product": "AUTO_PREMIUM", "limit": 3},
            )
            assert response.status_code == 200 and response.json() == {"results": []}
            response = client.post("/v1/knowledge/answer", json={"question": "test", "source_type": "PROCEDURE"})
            assert response.status_code == 200 and response.json()["insufficient_evidence"]
    assert loaded == ["retriever", "rag"] and search.closed
    assert search.calls[0]["filters"].product == "AUTO_PREMIUM"


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/v1/knowledge/search", {"query": "", "source_type": "POLICY"}),
        ("/v1/knowledge/search", {"query": "  ", "source_type": "POLICY"}),
        ("/v1/knowledge/search", {"query": "test", "source_type": "OTHER"}),
        ("/v1/knowledge/search", {"query": "test", "source_type": "POLICY", "limit": 21}),
        ("/v1/knowledge/search", {"query": "test", "source_type": "POLICY", "limit": True}),
        ("/v1/knowledge/search", {"query": "test", "source_type": "POLICY", "limit": 1.5}),
        ("/v1/knowledge/search", {"query": "test", "source_type": "POLICY", "as_of_date": "bad-date"}),
        ("/v1/knowledge/answer", {"question": "test", "source_type": "POLICY", "retrieval_limit": 0}),
        ("/v1/claims/demo/knowledge/search", {"query": "test", "source_type": "POLICY", "product": "AUTO_BASIC"}),
        ("/v1/claims/demo/knowledge/answer", {"question": "test", "source_type": "POLICY", "product": "AUTO_BASIC"}),
        ("/v1/claims/demo/knowledge/search", {"query": "test", "source_type": "POLICY", "as_of_date": "2027-01-01"}),
    ],
)
def test_invalid_payload_422_before_loader(path, payload):
    def forbidden():
        raise AssertionError("Invalid payload must not load services")

    with TestClient(create_app(AppConfig(knowledge_retriever_loader=forbidden))) as client:
        assert client.post(path, json=payload).status_code == 422
