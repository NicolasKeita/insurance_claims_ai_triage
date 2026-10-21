"""Test indexing."""

import pytest

from knowledge.indexing import KnowledgeIndexer
from knowledge.models import KnowledgeSourceType as Kind
from retrieval.models import IndexCompatibilityError
from tests.knowledge.index._support import store
from tests.knowledge.index._support import store as store


def test_index_separation_idempotence_and_manifest(store, tmp_path):
    indexer = KnowledgeIndexer(store)
    manifest = indexer.index_all(tmp_path / "manifest.json")
    counts = {k: store.count(k) for k in Kind}
    second = indexer.index_all(tmp_path / "manifest.json")
    assert counts == {k: store.count(k) for k in Kind}
    assert all(c["number_of_chunks"] == c["point_count"] for c in manifest["collections"])
    assert all(c["embedding_dimension"] == 8 and c["source_versions"] for c in second["collections"])
    for kind in Kind:
        records, _ = store.client.scroll(store.config.collection(kind), limit=100)
        assert all(r.payload["source_type"] == kind.value for r in records)
        assert all(r.payload["effective_from"] == "2026-01-01" for r in records)
    assert (tmp_path / "manifest.json").stat().st_size > 0


def test_contract_mismatch_no_automatic_delete_and_explicit_rebuild(store):
    store.ensure_collection(Kind.POLICY)
    contract = store.contract(Kind.POLICY) | {"embedding_model": "other-model-same-dimension"}
    store.client.update_collection(store.config.policy_collection, metadata={"knowledge_rag": contract})
    with pytest.raises(IndexCompatibilityError):
        store.ensure_collection(Kind.POLICY)
    assert store.client.collection_exists(store.config.policy_collection)
    store.ensure_collection(Kind.POLICY, recreate=True)
    store.require_collection(Kind.POLICY)
    store.client.update_collection(store.config.policy_collection, metadata={"knowledge_rag": None})
    with pytest.raises(IndexCompatibilityError, match="not owned"):
        store.ensure_collection(Kind.POLICY, recreate=True)
