from datetime import date
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from knowledge.config import ROOT, KNOWLEDGE_CHUNKING_VERSION, KnowledgeConfig
from knowledge.indexing import KnowledgeIndexer
from knowledge.models import KnowledgeSource, KnowledgeSourceType as Kind, applicable_on
from knowledge.qdrant_store import QdrantKnowledgeStore
from knowledge.retrieval import KnowledgeRetriever
from knowledge.sources import KnowledgeCatalog, chunk_markdown
from knowledge_fakes import FakeEmbeddingProvider
from retrieval.embeddings import SentenceTransformerEmbeddingProvider
from retrieval.models import IndexCompatibilityError, RetrievalUnavailableError


@pytest.fixture
def store():
    store = QdrantKnowledgeStore(QdrantClient(":memory:"), FakeEmbeddingProvider(), KnowledgeConfig())
    try:
        yield store
    finally:
        store.close()


def source(**updates):
    data = dict(source_id="TEST", source_type="POLICY", title="Demo", version="1",
                product="AUTO_PREMIUM", language="en", file_path="policies/test.md",
                effective_from=date(2026, 1, 1), effective_to=None)
    return KnowledgeSource(**(data | updates))


def test_chunking_stable_heading_paragraph_size_and_metadata():
    text = "# Demo\n\nIntro.\n\n## Collision Damage\n\n" + "A vehicle hits an object. " * 20 + "\n\n### Evidence\n\nPhotographs are needed."
    first = chunk_markdown(source(), text, 100)
    assert first == chunk_markdown(source(), text, 100)
    assert len({c.chunk_id for c in first}) == len(first)
    assert all(0 < len(c.text) <= 100 for c in first)
    assert all(c.chunking_version == KNOWLEDGE_CHUNKING_VERSION and c.product == "AUTO_PREMIUM" for c in first)
    assert first[-1].section == "Collision Damage / Evidence"
    assert all(c.source_version == "1" and c.effective_from == date(2026, 1, 1) for c in first)
    assert first != chunk_markdown(source(version="2"), text, 100)
    assert first != chunk_markdown(source(), text, 150)
    assert chunk_markdown(source(), "# Title\n\n ") == []
    long_token = chunk_markdown(source(), "## Token\n\n" + "a" * 250, 100)
    assert [len(c.text) for c in long_token] == [100, 100, 50]


def test_catalog_demo_and_validation(tmp_path):
    catalog = KnowledgeCatalog.load(ROOT / "data/knowledge")
    assert len(catalog.sources) == 3
    assert {s.source_type for s in catalog.sources} == {Kind.POLICY, Kind.PROCEDURE}
    assert len(catalog.chunks) > 10
    (tmp_path / "policies").mkdir()
    path = tmp_path / "policies/test.md"
    path.write_text("Not a synthetic source", encoding="utf-8")
    metadata = source().model_dump(mode="json", exclude={"file_path"})
    import json
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic"):
        KnowledgeCatalog.load(tmp_path)
    with pytest.raises(ValueError, match="No demo"):
        KnowledgeCatalog.load(tmp_path / "absent")


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


def test_retrieval_filters_limit_rank_authoritative_text_and_stale_chunks(store, tmp_path):
    KnowledgeIndexer(store).index_all(tmp_path / "manifest.json")
    service = KnowledgeRetriever(store)
    catalog = KnowledgeCatalog.load(store.config.knowledge_dir)
    target = next(c for c in catalog.chunks.values() if c.source_id == "AUTO_PREMIUM_POLICY" and c.section == "Deductible")
    response = service.search(target.text, Kind.POLICY, product="AUTO_PREMIUM", language="en", limit=3)
    assert len(response) == 3 and response[0].chunk_id == target.chunk_id
    assert [r.similarity_score for r in response] == sorted((r.similarity_score for r in response), reverse=True)
    assert all(r.product == "AUTO_PREMIUM" and r.source_type == Kind.POLICY for r in response)
    assert service.search("test", Kind.POLICY, language="fr") == []
    assert service.search("test", Kind.POLICY, product="NONEXISTENT") == []
    selected = service.search("test", Kind.POLICY, source_id="AUTO_BASIC_POLICY")
    assert selected and all(r.source_id == "AUTO_BASIC_POLICY" for r in selected)
    assert service.search("test", Kind.POLICY, product="AUTO_PREMIUM", source_id="AUTO_BASIC_POLICY") == []
    assert service.search("test", Kind.POLICY, as_of_date=date(2025, 12, 31)) == []
    assert all(r.source_type == Kind.PROCEDURE and r.product is None for r in service.search("test", Kind.PROCEDURE))
    # Do not expose forged payload text or provenance.
    store.client.set_payload(store.config.policy_collection, payload={"text": "FORGED", "title": "FORGED"}, points=[target.chunk_id])
    resolved = service.search(target.text, Kind.POLICY, product="AUTO_PREMIUM", limit=1)[0]
    assert resolved.text == target.text and resolved.title == target.title
    # A foreign ID cannot resolve to a local chunk; refill to requested limit.
    from uuid import uuid4
    foreign_id = str(uuid4())
    store.client.upsert(store.config.policy_collection, points=[models.PointStruct(
        id=foreign_id, vector=store.embeddings.embed_query(target.text),
        payload=target.model_dump(mode="json") | {"chunk_id": foreign_id},
    )])
    assert len(service.search(target.text, Kind.POLICY, product="AUTO_PREMIUM", limit=5)) == 5


def test_temporal_pure_and_qdrant_boundaries(store):
    bounded = source(effective_to=date(2026, 10, 1))
    assert not applicable_on(bounded, date(2025, 12, 31))
    assert applicable_on(bounded, date(2026, 1, 1))
    assert applicable_on(bounded, date(2026, 9, 30))
    assert not applicable_on(bounded, date(2026, 10, 1))
    assert applicable_on(bounded, None)
    assert applicable_on(source(effective_from=None), date(1900, 1, 1))
    with pytest.raises(ValueError, match="exclusive"):
        source(effective_to=date(2026, 1, 1))
    chunks = []
    for s in (bounded, source(version="2", effective_from=date(2026, 10, 1)),
              source(source_id="TIMELESS", effective_from=None)):
        chunks.extend(chunk_markdown(s, "## Rules\n\nA demo rule."))
    store.ensure_collection(Kind.POLICY)
    store.upsert(Kind.POLICY, chunks, store.embeddings.embed_documents([c.text for c in chunks]))
    for day, expected in ((date(2026, 1, 1), {chunks[0].chunk_id, chunks[2].chunk_id}),
                          (date(2026, 10, 1), {chunks[1].chunk_id, chunks[2].chunk_id}),
                          (date(2025, 1, 1), {chunks[2].chunk_id})):
        hits, _ = store.query(Kind.POLICY, store.embeddings.embed_query("rule"), as_of_date=day, limit=20)
        assert {i for i, _ in hits} == expected


def test_recreate_preflight_preserves_owned_collection_if_other_collection_is_foreign(store, tmp_path):
    store.ensure_collection(Kind.POLICY)
    chunk = chunk_markdown(source(), "## Rules\n\nA demo rule.")[0]
    store.upsert(Kind.POLICY, [chunk], store.embeddings.embed_documents([chunk.text]))
    store.client.create_collection(store.config.procedure_collection,
                                   vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE))
    with pytest.raises(IndexCompatibilityError, match="not owned"):
        KnowledgeIndexer(store).index_all(tmp_path / "manifest.json", recreate=True)
    assert store.count(Kind.POLICY) == 1


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


def test_asymmetric_encode_methods_and_legacy_fallback(monkeypatch):
    calls = []

    class Model:
        def get_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            calls.append("encode")
            assert kwargs["normalize_embeddings"]
            return np.array([[1, 0] for _ in texts])

        def encode_query(self, texts, **kwargs):
            calls.append("query")
            return self.encode(texts, **kwargs)

        def encode_document(self, texts, **kwargs):
            calls.append("document")
            return self.encode(texts, **kwargs)

    provider = SentenceTransformerEmbeddingProvider("unused", asymmetric=True)
    provider._model = Model()
    assert provider.embed_documents(["chunk"]) == [[1, 0]]
    assert provider.embed_query("question") == [1, 0]
    assert calls == ["document", "encode", "query", "encode"]
    provider._model.encode_document = None
    provider._model.encode_query = None
    assert provider.embed_documents(["chunk"]) == [[1, 0]]
    assert provider.embed_query("question") == [1, 0]
    assert provider.embed_documents([]) == []


def test_rag_model_config_independent_and_import_lazy(monkeypatch):
    monkeypatch.setenv("CLAIMS_EMBEDDING_MODEL", "claims-only")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "rag-only")
    assert KnowledgeConfig.from_env().embedding_model == "rag-only"
    with pytest.raises(ValueError, match="separate"):
        KnowledgeConfig(policy_collection="same", procedure_collection="same")
    with pytest.raises(ValueError, match="Similar Claims"):
        KnowledgeConfig(policy_collection="insurance_claims_similar_v1")
    result = subprocess.run([sys.executable, "-c", "import sys; import api.main; from knowledge.factory import create_knowledge_retriever; r = create_knowledge_retriever(); assert r.store.embeddings._model is None; assert 'sentence_transformers' not in sys.modules; r.close()"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
