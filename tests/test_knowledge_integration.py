"""Opt-in real Qdrant + Sentence Transformer retrieval, never Ollama."""

from dataclasses import replace
import os
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from knowledge.config import ROOT, KnowledgeConfig
from knowledge.evaluation import evaluate_retrieval, load_benchmark
from knowledge.indexing import KnowledgeIndexer
from knowledge.models import KnowledgeSourceType as Kind
from knowledge.qdrant_store import PAYLOAD_INDEXES, QdrantKnowledgeStore
from knowledge.retrieval import KnowledgeRetriever
from retrieval.embeddings import SentenceTransformerEmbeddingProvider

pytestmark = pytest.mark.rag


def test_real_retrieval_index_filters_and_complete_benchmark(tmp_path):
    if os.environ.get("RUN_RAG_TESTS") != "1" or not os.environ.get("QDRANT_URL"):
        pytest.skip("Set RUN_RAG_TESTS=1 and QDRANT_URL for real retrieval integration")
    suffix = uuid4().hex
    config = replace(KnowledgeConfig.from_env(), policy_collection=f"test_rag_policy_{suffix}",
                     procedure_collection=f"test_rag_procedure_{suffix}")
    client = QdrantClient(url=config.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=10)
    embeddings = SentenceTransformerEmbeddingProvider(config.embedding_model, asymmetric=True)
    store = QdrantKnowledgeStore(client, embeddings, config)
    try:
        first = KnowledgeIndexer(store).index_all(tmp_path / "manifest.json")
        KnowledgeIndexer(store).index_all(tmp_path / "manifest.json")
        for kind, entry in zip(Kind, first["collections"], strict=True):
            assert store.count(kind) == entry["number_of_chunks"]
            assert set(PAYLOAD_INDEXES) <= set(client.get_collection(config.collection(kind)).payload_schema)
        retriever = KnowledgeRetriever(store)
        results = retriever.search("collision deductible", Kind.POLICY, product="AUTO_PREMIUM", limit=3)
        assert results and all(r.product == "AUTO_PREMIUM" for r in results)
        assert len(results) <= 3
        assert retriever.search("test", Kind.POLICY, language="fr") == []
        assert all(r.source_type == Kind.PROCEDURE for r in retriever.search("missing documents", Kind.PROCEDURE))
        report = evaluate_retrieval(retriever, load_benchmark(ROOT / "data/evaluation/rag_retrieval_v1.json"))
        assert report["evaluated_cases"] == 13
        assert all(0 <= report[k] <= 1 for k in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr"))
    finally:
        # Only this test's random, reserved collections may be cleaned up.
        for kind in Kind:
            if client.collection_exists(config.collection(kind)):
                client.delete_collection(config.collection(kind))
        client.close()
