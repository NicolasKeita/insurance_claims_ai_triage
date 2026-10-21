"""Test errors."""

from fastapi.testclient import TestClient

from api.main import AppConfig, create_app
from knowledge.models import GroundingError
from retrieval.models import RetrievalUnavailableError
from tests.knowledge.api._support import FakeRag, FakeSearch


def test_clean_errors_and_database_unconfigured(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with TestClient(create_app()) as client:
        response = client.post("/v1/claims/demo/knowledge/search", json={"query": "test", "source_type": "POLICY"})
        assert response.status_code == 503 and response.json()["detail"] == "Database is not configured"

    class BrokenSearch(FakeSearch):
        def search(self, *args, **kwargs):
            raise RetrievalUnavailableError("Knowledge index is missing")

    with TestClient(create_app(AppConfig(knowledge_retriever_loader=lambda: BrokenSearch()))) as client:
        assert client.post("/v1/knowledge/search", json={"query": "test", "source_type": "POLICY"}).status_code == 503

    class BrokenRag(FakeRag):
        def answer(self, *args, **kwargs):
            raise GroundingError("Unknown citation")

    with TestClient(
        create_app(
            AppConfig(
                knowledge_retriever_loader=lambda: FakeSearch(),
                knowledge_rag_loader=lambda _: BrokenRag(),
            )
        )
    ) as client:
        assert (
            client.post("/v1/knowledge/answer", json={"question": "test", "source_type": "POLICY"}).status_code == 502
        )
