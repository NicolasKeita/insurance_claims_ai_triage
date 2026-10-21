"""Test refill."""

from retrieval.representation import (
    build_retrieval_document,
)
from retrieval.similar_claims import SimilarClaimsIndexer, SimilarClaimsService
from tests.infrastructure.support.retrieval_fakes import FakeRepository, claim
from tests.intelligence.retrieval.core._support import components
from tests.intelligence.retrieval.core._support import components as components


def test_search_refills_after_a_whole_page_of_deleted_points(components):
    """
    Keep only the last ranked candidate; determine ranking before removing rows.
    """
    embeddings, store = components
    repository = FakeRepository([claim(), *[claim(i, days=1) for i in range(2, 55)]])
    SimilarClaimsIndexer(repository, embeddings, store).index_all()
    current = repository.get("DEMO-TEST-1")
    vector = embeddings.embed_query(build_retrieval_document(current).canonical_text)
    ordered = store.query(vector, current.incident_date, current.facts.claim_type, 100).hits
    last = repository.claims[ordered[-1].point_id]
    repository.claims = {current.point_id: current, last.point_id: last}
    response = SimilarClaimsService(repository, embeddings, store).search(current.claim_id, 1)
    assert [r.claim_id for r in response.results] == [last.claim_id]
    assert len(repository.bulk_reads) == 3
