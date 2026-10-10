from datetime import date
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from api.main import create_app
from knowledge.models import GroundedAnswer, GroundingError, KnowledgeSourceType as Kind
from persistence.repositories import ClaimNotFoundError
from retrieval.models import RetrievalUnavailableError


class FakeSearch:
    def __init__(self):
        self.calls = []
        self.closed = False

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return []

    def close(self):
        self.closed = True


class FakeRag:
    def __init__(self):
        self.calls = []

    def answer(self, **kwargs):
        self.calls.append(kwargs)
        return GroundedAnswer(answer="Demo excerpts are insufficient.", citations=[], insufficient_evidence=True)


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

    with TestClient(create_app(knowledge_retriever_loader=loader, knowledge_rag_loader=rag_loader)) as client:
        assert loaded == []
        for _ in range(2):
            response = client.post("/v1/knowledge/search", json={"query": "test", "source_type": "POLICY", "product": "AUTO_PREMIUM", "limit": 3})
            assert response.status_code == 200 and response.json() == {"results": []}
            response = client.post("/v1/knowledge/answer", json={"question": "test", "source_type": "PROCEDURE"})
            assert response.status_code == 200 and response.json()["insufficient_evidence"]
    assert loaded == ["retriever", "rag"] and search.closed
    assert search.calls[0]["product"] == "AUTO_PREMIUM"


@pytest.mark.parametrize("path,payload", [
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
])
def test_invalid_payload_422_before_loader(path, payload):
    def forbidden():
        raise AssertionError("Invalid payload must not load services")

    with TestClient(create_app(knowledge_retriever_loader=forbidden)) as client:
        assert client.post(path, json=payload).status_code == 422


def test_claim_aware_product_and_dates_and_unknown_before_model_io():
    search, rag = FakeSearch(), FakeRag()
    claim_calls = []

    def claim_loader(factory, claim_id):
        claim_calls.append(claim_id)
        if claim_id == "missing":
            raise ClaimNotFoundError("Missing claim")
        return SimpleNamespace(policy=SimpleNamespace(product="AUTO_PREMIUM"), incident=SimpleNamespace(date=date(2026, 9, 20)))

    app = create_app(knowledge_retriever_loader=lambda: search, knowledge_rag_loader=lambda _: rag,
                     knowledge_claim_loader=claim_loader)
    with TestClient(app) as client:
        for kind in ("POLICY", "PROCEDURE"):
            for operation, field in (("search", "query"), ("answer", "question")):
                assert client.post(f"/v1/claims/demo/knowledge/{operation}", json={field: "test", "source_type": kind}).status_code == 200
        assert client.post("/v1/claims/missing/knowledge/search", json={"query": "test", "source_type": "POLICY"}).status_code == 404
    assert search.calls[0]["product"] == rag.calls[0]["product"] == "AUTO_PREMIUM"
    assert search.calls[0]["as_of_date"] == rag.calls[0]["as_of_date"] == date(2026, 9, 20)
    assert search.calls[1]["product"] is rag.calls[1]["product"] is None
    assert search.calls[1]["as_of_date"] is rag.calls[1]["as_of_date"] is None
    assert len(search.calls) == 2 and len(claim_calls) == 5


def test_clean_errors_and_database_unconfigured(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with TestClient(create_app()) as client:
        response = client.post("/v1/claims/demo/knowledge/search", json={"query": "test", "source_type": "POLICY"})
        assert response.status_code == 503 and response.json()["detail"] == "Database is not configured"

    class BrokenSearch(FakeSearch):
        def search(self, **kwargs):
            raise RetrievalUnavailableError("Knowledge index is missing")

    with TestClient(create_app(knowledge_retriever_loader=lambda: BrokenSearch())) as client:
        assert client.post("/v1/knowledge/search", json={"query": "test", "source_type": "POLICY"}).status_code == 503

    class BrokenRag(FakeRag):
        def answer(self, **kwargs):
            raise GroundingError("Unknown citation")

    with TestClient(create_app(knowledge_retriever_loader=lambda: FakeSearch(), knowledge_rag_loader=lambda _: BrokenRag())) as client:
        assert client.post("/v1/knowledge/answer", json={"question": "test", "source_type": "POLICY"}).status_code == 502
