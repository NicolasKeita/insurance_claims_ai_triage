"""Select with ``pytest -m rag``: real Qdrant and a cached embedding model."""

from dataclasses import replace

import pytest
from integration_support import get_test_qdrant_url
from qdrant_integration_support import temporary_qdrant_collections

from knowledge.config import ROOT, KnowledgeConfig
from knowledge.evaluation import evaluate_retrieval, load_benchmark
from knowledge.indexing import KnowledgeIndexer
from knowledge.models import KnowledgeFilter
from knowledge.models import KnowledgeSourceType as Kind
from knowledge.qdrant_store import PAYLOAD_INDEXES, QdrantKnowledgeStore
from knowledge.retrieval import KnowledgeRetriever
from retrieval.embeddings import SentenceTransformerEmbeddingProvider

pytestmark = [pytest.mark.integration, pytest.mark.rag]


def test_real_retrieval_index_filters_and_complete_benchmark(tmp_path, monkeypatch):
    """
    Never trigger a Hub request, even when other tests imported Transformers first.

    Fixed, tiny synthetic corpus only; app connection/collection overrides are ignored.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    with temporary_qdrant_collections("rag_policy", "rag_procedure") as (client, collections):
        config = replace(
            KnowledgeConfig(),
            qdrant_url=get_test_qdrant_url(),
            policy_collection=collections[0],
            procedure_collection=collections[1],
            knowledge_dir=ROOT / "data" / "knowledge",
        )
        embeddings = SentenceTransformerEmbeddingProvider(
            config.embedding_model, asymmetric=True, local_files_only=True
        )
        try:
            assert embeddings.dimension > 0
        except Exception as error:
            pytest.fail(
                f"RAG requires the locally cached embedding model {config.embedding_model!r}. "
                "No model is downloaded by pytest; preload it explicitly before running 'pytest -m rag'. "
                f"Original error: {type(error).__name__}: {error}",
                pytrace=False,
            )
        store = QdrantKnowledgeStore(client, embeddings, config)
        first = KnowledgeIndexer(store).index_all(tmp_path / "manifest.json")
        KnowledgeIndexer(store).index_all(tmp_path / "manifest.json")
        for kind, entry in zip(Kind, first["collections"], strict=True):
            assert store.count(kind) == entry["number_of_chunks"]
            assert set(PAYLOAD_INDEXES) <= set(client.get_collection(config.collection(kind)).payload_schema)
        retriever = KnowledgeRetriever(store)
        results = retriever.search(
            "collision deductible", Kind.POLICY, filters=KnowledgeFilter(product="AUTO_PREMIUM"), limit=3
        )
        assert results and all(r.product == "AUTO_PREMIUM" for r in results)
        assert len(results) <= 3
        assert retriever.search("test", Kind.POLICY, filters=KnowledgeFilter(language="fr")) == []
        assert all(r.source_type == Kind.PROCEDURE for r in retriever.search("missing documents", Kind.PROCEDURE))
        report = evaluate_retrieval(retriever, load_benchmark(ROOT / "data/evaluation/rag_retrieval_v1.json"))
        assert report["evaluated_cases"] == 13
        assert all(0 <= report[k] <= 1 for k in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr"))
