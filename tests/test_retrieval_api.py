from fastapi.testclient import TestClient
import pytest

from api.main import create_app
from persistence.repositories import ClaimNotFoundError
from retrieval.models import RetrievalUnavailableError, SimilarClaimsResponse
from retrieval.representation import CLAIM_RETRIEVAL_REPRESENTATION_VERSION


def forbidden():
    raise AssertionError("Other API routes must not load the real embedding model")


def test_similar_service_cached_and_unknown_404():
    calls = []

    class FakeService:
        def search(self, claim_id, limit):
            if claim_id == "missing":
                raise ClaimNotFoundError("Claim missing does not exist")
            calls.append((claim_id, limit))
            return SimilarClaimsResponse(claim_id=claim_id, representation_version=CLAIM_RETRIEVAL_REPRESENTATION_VERSION,
                                         results=[])

    loaded = []

    def loader(factory):
        loaded.append(factory)
        return FakeService()

    with TestClient(create_app(forbidden, forbidden, similar_service_loader=loader)) as client:
        for limit in (1, 5, 20):
            response = client.get(f"/v1/claims/demo/similar?limit={limit}")
            assert response.status_code == 200
            assert response.json() == {"claim_id": "demo", "representation_version": "claim_retrieval_v1", "results": []}
        assert client.get("/v1/claims/missing/similar").status_code == 404
    assert len(loaded) == 1
    assert calls == [("demo", 1), ("demo", 5), ("demo", 20)]


@pytest.mark.parametrize("limit", ["0", "-1", "21", "1.5", "abc"])
def test_api_rejects_invalid_limit_without_service_load(limit):
    with TestClient(create_app(similar_service_loader=lambda _: forbidden())) as client:
        assert client.get(f"/v1/claims/demo/similar?limit={limit}").status_code == 422


def test_api_index_unavailable_is_clean_503():
    class BrokenService:
        def search(self, *_):
            raise RetrievalUnavailableError("Similar claims index is unavailable")

    with TestClient(create_app(similar_service_loader=lambda _: BrokenService())) as client:
        response = client.get("/v1/claims/demo/similar")
        assert response.status_code == 503
        assert response.json() == {"detail": "Similar claims index is unavailable"}


def test_api_database_unconfigured(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with TestClient(create_app()) as client:
        response = client.get("/v1/claims/demo/similar")
        assert response.status_code == 503
        assert response.json() == {"detail": "Database is not configured"}
