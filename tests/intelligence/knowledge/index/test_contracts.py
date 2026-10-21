"""Test contracts."""

import pytest

from retrieval.knowledge.models import KnowledgeSourceType as Kind
from retrieval.knowledge.qdrant_store import QdrantKnowledgeStore
from retrieval.knowledge.retrieval import KnowledgeRetriever
from retrieval.knowledge.sources import chunk_markdown
from retrieval.models import IndexCompatibilityError, RetrievalUnavailableError
from tests.intelligence.knowledge.index._support import source, store
from tests.intelligence.knowledge.index._support import store as store


def test_chunking_and_dimension_contract_mismatches(store):
    store.ensure_collection(Kind.POLICY)
    from dataclasses import replace

    changed = QdrantKnowledgeStore(store.client, store.embeddings, replace(store.config, max_chunk_chars=700))
    with pytest.raises(IndexCompatibilityError):
        changed.require_collection(Kind.POLICY)
    dimension = store.embeddings.dimension
    store.embeddings.dimension = 9
    with pytest.raises(IndexCompatibilityError):
        store.require_collection(Kind.POLICY)
    store.embeddings.dimension = dimension


@pytest.mark.parametrize("limit", [0, -1, 21, 1.5, True])
def test_invalid_limit_before_model_io(store, limit):
    with pytest.raises(ValueError, match="limit"):
        KnowledgeRetriever(store).search("test", Kind.POLICY, limit=limit)


def test_missing_collection_read_only_and_invalid_vectors(store):
    with pytest.raises(RetrievalUnavailableError, match="missing"):
        KnowledgeRetriever(store).search("test", Kind.POLICY)
    assert not store.client.collection_exists(store.config.policy_collection)
    chunk = chunk_markdown(source(), "## Rules\n\nRule.")[0]
    store.ensure_collection(Kind.POLICY)
    for vector in ([0] * 8, [2] * 8, [1], [float("nan")] * 8):
        with pytest.raises(ValueError, match="normalized"):
            store.upsert(Kind.POLICY, [chunk], [vector])
    with pytest.raises(ValueError, match="source type"):
        store.upsert(Kind.PROCEDURE, [chunk], store.embeddings.embed_documents([chunk.text]))
