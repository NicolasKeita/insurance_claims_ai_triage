"""Explicitly opt-in real Qdrant; each test owns a unique derived collection."""

import os
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from retrieval.qdrant_store import QdrantClaimStore
from retrieval.similar_claims import SimilarClaimsIndexer, SimilarClaimsService
from retrieval_fakes import FakeEmbeddings, FakeRepository, claim

pytestmark = pytest.mark.qdrant


@pytest.fixture
def remote_store():
    if os.environ.get("RUN_QDRANT_TESTS") != "1" or not os.environ.get("QDRANT_URL"):
        pytest.skip("Set RUN_QDRANT_TESTS=1 and QDRANT_URL for real Qdrant tests")
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"), timeout=10)
    collection = f"test_similar_claims_{uuid4().hex}"
    store = QdrantClaimStore(client, collection, FakeEmbeddings())
    try:
        yield store
    finally:
        # Only delete the unique collection this fixture reserved, never app data.
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.close()


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
