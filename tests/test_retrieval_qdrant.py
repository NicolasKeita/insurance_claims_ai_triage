"""Select with ``pytest -m qdrant``; each test owns a unique collection."""

import pytest

from retrieval.qdrant_store import QdrantClaimStore
from retrieval.similar_claims import SimilarClaimsIndexer, SimilarClaimsService
from retrieval_fakes import FakeEmbeddings, FakeRepository, claim
from qdrant_integration_support import temporary_qdrant_collections

pytestmark = [pytest.mark.integration, pytest.mark.qdrant]


@pytest.fixture
def remote_store():
    with temporary_qdrant_collections("similar_claims") as (client, collections):
        yield QdrantClaimStore(client, collections[0], FakeEmbeddings())


def test_real_qdrant_idempotence_filtering_and_contract(remote_store):
    claims = [claim(), claim(2, days=1), claim(3, days=0), claim(4, days=-1), claim(5, days=2, claim_type="HOME")]
    repo = FakeRepository(claims)
    indexer = SimilarClaimsIndexer(repo, remote_store.embeddings, remote_store, batch_size=2)
    indexer.index_all()
    indexer.index_all()
    assert remote_store.count() == 5
    info = remote_store.client.get_collection(remote_store.collection)
    assert {"incident_date_ordinal", "claim_type", "representation_version"} <= set(info.payload_schema)
    assert info.config.metadata["claim_retrieval"] == remote_store.contract()
    results = SimilarClaimsService(repo, remote_store.embeddings, remote_store).search(claims[0].claim_id).results
    assert [r.claim_id for r in results] == [claims[1].claim_id]
