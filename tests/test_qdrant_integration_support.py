"""Connection and cleanup failures must be explicit without touching app data."""

import sys
from types import SimpleNamespace

import pytest

import qdrant_integration_support as support
from retrieval.embeddings import SentenceTransformerEmbeddingProvider


@pytest.fixture
def client_stub(monkeypatch):
    class Client:
        def __init__(self):
            self.collections = {"insurance_claims_similar_v1"}
            self.deleted = []
            self.closed = False

        def get_collections(self):
            return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.collections])

        def collection_exists(self, name):
            return name in self.collections

        def delete_collection(self, name):
            self.deleted.append(name)
            self.collections.remove(name)

        def close(self):
            self.closed = True

    client = Client()
    monkeypatch.setattr(support, "QdrantClient", lambda **options: client)
    monkeypatch.setattr(support, "get_test_qdrant_url", lambda: "http://127.0.0.1:6335")
    return client


def test_failed_test_still_cleans_own_collections_and_preserves_application(client_stub):
    with pytest.raises(AssertionError, match="test failure"):
        with support.temporary_qdrant_collections("rag_policy", "rag_procedure") as (client, names):
            assert len(set(names)) == 2
            assert all(name.startswith("test_rag_") for name in names)
            client.collections.update(names)
            raise AssertionError("test failure")
    assert set(client_stub.deleted) == set(names)
    assert client_stub.collections == {"insurance_claims_similar_v1"}
    assert client_stub.closed


def test_cleanup_attempts_every_collection_and_closes_after_delete_error(client_stub, monkeypatch):
    attempted = []

    def fail_first_delete(name):
        attempted.append(name)
        if len(attempted) == 1:
            raise ConnectionError("cleanup unavailable")
        client_stub.collections.remove(name)

    monkeypatch.setattr(client_stub, "delete_collection", fail_first_delete)
    with pytest.raises(ExceptionGroup, match="Test Qdrant cleanup failed"):
        with support.temporary_qdrant_collections("rag_policy", "rag_procedure") as (client, names):
            client.collections.update(names)
    assert attempted == list(names)
    assert client_stub.closed
    assert "insurance_claims_similar_v1" in client_stub.collections


def test_unreachable_service_fails_with_start_command_and_closes(client_stub, monkeypatch):
    def unavailable():
        raise ConnectionError("connection refused")

    monkeypatch.setattr(client_stub, "get_collections", unavailable)
    with pytest.raises(RuntimeError, match="Test Qdrant is unreachable.*qdrant-test.*TEST_QDRANT_URL"):
        with support.temporary_qdrant_collections("similar_claims"):
            pytest.fail("An unavailable service must fail before yielding")
    assert client_stub.closed
    assert not client_stub.deleted


def test_existing_collection_collision_never_deleted(client_stub, monkeypatch):
    monkeypatch.setattr(support, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    client_stub.collections.add(f"test_similar_claims_{'a' * 32}")
    before = set(client_stub.collections)
    with pytest.raises(RuntimeError, match="Refusing to reserve an existing Qdrant collection"):
        with support.temporary_qdrant_collections("similar_claims"):
            pytest.fail("Existing data must never be reserved by a test")
    assert client_stub.collections == before
    assert not client_stub.deleted
    assert client_stub.closed


def test_offline_embedding_load_is_lazy_and_missing_cache_cannot_download(monkeypatch):
    calls = []

    def missing_model(name, **options):
        calls.append((name, options))
        raise OSError("model absent from local cache")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=missing_model))
    provider = SentenceTransformerEmbeddingProvider("uncached/test-model", local_files_only=True)
    assert calls == []
    with pytest.raises(OSError, match="absent from local cache"):
        _ = provider.dimension
    assert calls == [("uncached/test-model", {"local_files_only": True})]
