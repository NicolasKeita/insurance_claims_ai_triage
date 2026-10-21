"""Test scope."""

from datetime import date
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import AppConfig, create_app
from persistence.repositories import ClaimNotFoundError
from tests.intelligence.knowledge.api._support import FakeRag, FakeSearch


def test_claim_aware_product_and_dates_and_unknown_before_model_io():
    search, rag = FakeSearch(), FakeRag()
    claim_calls = []

    def claim_loader(factory, claim_id):
        claim_calls.append(claim_id)
        if claim_id == "missing":
            raise ClaimNotFoundError("Missing claim")
        return SimpleNamespace(
            policy=SimpleNamespace(product="AUTO_PREMIUM"), incident=SimpleNamespace(date=date(2026, 9, 20))
        )

    app = create_app(
        AppConfig(
            knowledge_retriever_loader=lambda: search,
            knowledge_rag_loader=lambda _: rag,
            knowledge_claim_loader=claim_loader,
        )
    )
    with TestClient(app) as client:
        for kind in ("POLICY", "PROCEDURE"):
            for operation, field in (("search", "query"), ("answer", "question")):
                assert (
                    client.post(
                        f"/v1/claims/demo/knowledge/{operation}", json={field: "test", "source_type": kind}
                    ).status_code
                    == 200
                )
        assert (
            client.post(
                "/v1/claims/missing/knowledge/search", json={"query": "test", "source_type": "POLICY"}
            ).status_code
            == 404
        )
    assert search.calls[0]["filters"].product == rag.calls[0]["filters"].product == "AUTO_PREMIUM"
    assert search.calls[0]["filters"].as_of_date == rag.calls[0]["filters"].as_of_date == date(2026, 9, 20)
    assert search.calls[1]["filters"].product is rag.calls[1]["filters"].product is None
    assert search.calls[1]["filters"].as_of_date is rag.calls[1]["filters"].as_of_date is None
    assert len(search.calls) == 2 and len(claim_calls) == 5
