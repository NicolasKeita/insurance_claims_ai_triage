"""Test preflight."""

import pytest
from qdrant_client import models

from retrieval.knowledge.indexing import KnowledgeIndexer
from retrieval.knowledge.models import KnowledgeSourceType as Kind
from retrieval.knowledge.sources import chunk_markdown
from retrieval.models import IndexCompatibilityError
from tests.intelligence.knowledge.index._support import source, store
from tests.intelligence.knowledge.index._support import store as store


def test_recreate_preflight_preserves_owned_collection_if_other_collection_is_foreign(store, tmp_path):
    store.ensure_collection(Kind.POLICY)
    chunk = chunk_markdown(source(), "## Rules\n\nA demo rule.")[0]
    store.upsert(Kind.POLICY, [chunk], store.embeddings.embed_documents([chunk.text]))
    store.client.create_collection(
        store.config.procedure_collection, vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE)
    )
    with pytest.raises(IndexCompatibilityError, match="not owned"):
        KnowledgeIndexer(store).index_all(tmp_path / "manifest.json", recreate=True)
    assert store.count(Kind.POLICY) == 1
