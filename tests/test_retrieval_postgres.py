"""Real PostgreSQL boundary, query count, and opt-in combined Qdrant/API checks."""

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select

from api.main import create_app
from persistence.models import ClaimRow
from persistence.repositories import ClaimNotFoundError, ClaimRepository
from persistence.retrieval import RetrievalClaimRepository
from retrieval.models import StoredRetrievalClaim
from retrieval.representation import build_claim_retrieval_text
from retrieval.similar_claims import SimilarClaimsIndexer, SimilarClaimsService
from scripts.seed_similar_claims import demo_claims, seed
from test_persistence_postgres import factory, postgres_engine, reference_claim
from test_retrieval_qdrant import remote_store

pytestmark = pytest.mark.postgres


def test_repository_bulk_reads_two_queries_per_batch(factory, reference_claim):
    with factory.begin() as session:
        repo = ClaimRepository(session)
        repo.add(reference_claim)
        for demo in demo_claims(reference_claim):
            repo.add(demo)
    repository = RetrievalClaimRepository(factory)
    sql = []
    connection = factory.kw["bind"]

    def record(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            sql.append(statement)

    event.listen(connection, "before_cursor_execute", record)
    try:
        batches = list(repository.iter_batches(3))
        assert [len(batch) for batch in batches] == [3, 3, 1]
        assert len(sql) == 7  # Two per nonempty batch, one final empty read.
        assert all(isinstance(c, StoredRetrievalClaim) for b in batches for c in b)
        sql.clear()
        current = repository.get(reference_claim.claim_id)
        assert len(sql) == 2
        assert current.facts.repair_amount == Decimal("3160")
        assert "customer" not in build_claim_retrieval_text(current.facts)
        sql.clear()
        loaded = repository.get_many([c.point_id for b in batches for c in b])
        assert len(loaded) == 7
        assert len(sql) == 2
        assert repository.get_many([]) == {}
        with pytest.raises(ClaimNotFoundError):
            repository.get("missing")
    finally:
        event.remove(connection, "before_cursor_execute", record)


def test_step26_seed_idempotent_and_preserves_step25_history(factory, reference_claim):
    from persistence.history import HistoricalClaimRepository
    from scripts.seed_historical_claims import seed as seed_history

    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        seed_history(session)
        before = HistoricalClaimRepository(session).get_profile(reference_claim.claim_id)
        ids, created = seed(session)
        assert len(ids) == created == 6
        assert seed(session) == (ids, 0)
        assert HistoricalClaimRepository(session).get_profile(reference_claim.claim_id) == before


@pytest.mark.qdrant
def test_postgres_qdrant_and_api_use_authoritative_business_data(factory, reference_claim, remote_store):
    from scripts.seed_historical_claims import seed as seed_history

    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        seed_history(session)
        seed(session)
    repository = RetrievalClaimRepository(factory)
    indexer = SimilarClaimsIndexer(repository, remote_store.embeddings, remote_store, batch_size=3)
    assert indexer.index_all().points_upserted == 16
    assert indexer.index_all().points_upserted == 16
    assert remote_store.count() == 16
    service = SimilarClaimsService(repository, remote_store.embeddings, remote_store)
    initial = service.search(reference_claim.claim_id)
    assert len(initial.results) == 5
    assert all(r.incident_date < reference_claim.incident.date for r in initial.results)
    first = initial.results[0]
    with factory.begin() as session:
        row = session.scalar(select(ClaimRow).where(ClaimRow.claim_id == first.claim_id))
        row.repair_estimate_amount = Decimal("3200")
    # Same indexed vector/rank, fresh PostgreSQL display amount.
    class BorrowedService:
        search = service.search

    with TestClient(create_app(session_factory=factory, similar_service_loader=lambda _: BorrowedService())) as client:
        response = client.get(f"/v1/claims/{reference_claim.claim_id}/similar")
        assert response.status_code == 200
        assert response.json()["results"][0]["repair_amount"] == "3200"
        assert response.json()["results"][0]["similarity_score"] == first.similarity_score
        assert client.get("/v1/claims/missing/similar").status_code == 404
    with factory.begin() as session:
        row = session.scalar(select(ClaimRow).where(ClaimRow.claim_id == first.claim_id))
        row.incident_date = reference_claim.incident.date
    assert first.claim_id not in {r.claim_id for r in service.search(reference_claim.claim_id).results}
