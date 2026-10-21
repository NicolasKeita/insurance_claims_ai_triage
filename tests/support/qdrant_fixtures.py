import pytest

from retrieval.qdrant_store import QdrantClaimStore
from tests.support.qdrant_integration_support import temporary_qdrant_collections
from tests.support.retrieval_fakes import FakeEmbeddings


@pytest.fixture
def remote_store():
    with temporary_qdrant_collections("similar_claims") as (client, collections):
        yield QdrantClaimStore(client, collections[0], FakeEmbeddings())
