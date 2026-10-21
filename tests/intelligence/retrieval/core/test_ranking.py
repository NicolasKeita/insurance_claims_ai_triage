"""Test ranking."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

from retrieval.representation import (
    build_retrieval_document,
)
from retrieval.similar_claims import SimilarClaimsIndexer, SimilarClaimsService
from tests.infrastructure.support.retrieval_fakes import FakeRepository, claim
from tests.intelligence.retrieval.core._support import components
from tests.intelligence.retrieval.core._support import components as components


def test_ranking_top_k_and_strict_date_and_type_filters(components):
    """
    Qdrant itself excludes type/date/current; do not rely on app filtering alone.

    compares to actual HOME, never hardcodes AUTO_COLLISION
    """
    embeddings, store = components
    claims = [
        claim(),
        claim(2, days=2),
        claim(3, days=4, amount="3500"),
        claim(4, days=6, collision="PARKING_DAMAGE"),
        claim(5, days=0),
        claim(6, days=-1),
        claim(7, days=10, claim_type="HOME"),
    ]
    repository = FakeRepository(claims)
    SimilarClaimsIndexer(repository, embeddings, store).index_all()
    service = SimilarClaimsService(repository, embeddings, store)
    results = service.search(claims[0].claim_id, 2).results
    assert [r.claim_id for r in results] == [claims[1].claim_id, claims[2].claim_id]
    assert results[0].similarity_score > results[1].similarity_score
    results = service.search(claims[0].claim_id, 20).results
    assert [r.claim_id for r in results] == [c.claim_id for c in claims[1:4]]
    assert all(r.incident_date < claims[0].incident_date for r in results)
    page = store.query(
        embeddings.embed_query(build_retrieval_document(claims[0]).canonical_text),
        claims[0].incident_date,
        "AUTO_COLLISION",
        20,
    )
    assert {h.point_id for h in page.hits} == {c.point_id for c in claims[1:4]}
    other_domain = service.search(claims[-1].claim_id).results
    assert other_domain == []
    unfiltered = SimilarClaimsService(repository, embeddings, store, same_claim_type=False)
    assert claims[-1].claim_id in {r.claim_id for r in unfiltered.search(claims[0].claim_id).results}


def test_stale_points_ignored_and_postgres_facts_used(components):
    embeddings, store = components
    claims = [claim(), *[claim(i, days=i) for i in range(2, 7)]]
    repository = FakeRepository(claims)
    SimilarClaimsIndexer(repository, embeddings, store).index_all()
    del repository.claims[claims[1].point_id]
    repository.claims[claims[2].point_id] = replace(claims[2], incident_date=claims[0].incident_date)
    repository.claims[claims[3].point_id] = replace(
        claims[3], incident_date=claims[0].incident_date + timedelta(days=1)
    )
    repository.claims[claims[4].point_id] = replace(claims[4], facts=replace(claims[4].facts, claim_type="HOME"))
    repository.claims[claims[5].point_id] = replace(
        claims[5], facts=replace(claims[5].facts, repair_amount=Decimal("3200"), collision_type="SIDE_COLLISION")
    )
    results = SimilarClaimsService(repository, embeddings, store).search(claims[0].claim_id).results
    assert len(results) == 1
    assert results[0].claim_id == claims[5].claim_id
    assert results[0].repair_amount == Decimal("3200")
    assert results[0].collision_type == "SIDE_COLLISION"
