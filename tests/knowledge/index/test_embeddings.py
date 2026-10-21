"""Test embeddings."""

import subprocess
import sys

import numpy as np
import pytest

from knowledge.config import KnowledgeConfig
from retrieval.embeddings import SentenceTransformerEmbeddingProvider
from tests.knowledge.index._support import store as store


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
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import api.main; from knowledge.factory import create_knowledge_retriever; "
            "r = create_knowledge_retriever(); assert r.store.embeddings._model is None; "
            "assert 'sentence_transformers' not in sys.modules; r.close()",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
