"""Test embeddings."""

import subprocess
import sys
from types import SimpleNamespace

from retrieval.embeddings import SentenceTransformerEmbeddingProvider
from tests.intelligence.retrieval.core._support import components as components


def test_import_api_and_provider_construction_loads_no_sentence_transformer():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import api.main; from retrieval.embeddings import SentenceTransformerEmbeddingProvider; "
            "p = SentenceTransformerEmbeddingProvider('unused'); "
            "assert p._model is None; assert 'sentence_transformers' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_provider_loads_once_and_normalizes_symmetric_embeddings(monkeypatch):
    loads, calls = [], []

    class Model:
        def __init__(self, name):
            loads.append(name)

        def get_embedding_dimension(self):
            return 4

        def encode(self, texts, **kwargs):
            calls.append(kwargs)
            return [[1, 0, 0, 0] for _ in texts]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    provider = SentenceTransformerEmbeddingProvider("mock-model")
    assert loads == []
    assert provider.dimension == 4
    assert provider.embed_documents(["a", "b"]) == [[1, 0, 0, 0], [1, 0, 0, 0]]
    assert provider.embed_query("a") == [1, 0, 0, 0]
    assert provider.embed_documents([]) == []
    assert loads == ["mock-model"]
    assert all(call["normalize_embeddings"] for call in calls)


def test_encoding_modes_share_weights_without_changing_rag_or_claim_vectors(monkeypatch):
    loads, encodes = [], []

    class Model:
        def __init__(self, name, **options):
            loads.append((name, options))

        def encode(self, texts, **options):
            encodes.append("symmetric")
            return [[1, 0] for _ in texts]

        def encode_document(self, texts, **options):
            encodes.append("document")
            return [[0, 1] for _ in texts]

        def encode_query(self, texts, **options):
            encodes.append("query")
            return [[0, -1] for _ in texts]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    rag = SentenceTransformerEmbeddingProvider("shared", asymmetric=True, local_files_only=True)
    claims = rag.with_encoding_mode(asymmetric=False)
    assert rag._model is claims._model and rag is not claims
    assert rag.asymmetric is True and claims.asymmetric is False
    assert rag.embed_documents(["policy"]) == [[0, 1]]
    assert rag.embed_query("question") == [0, -1]
    assert claims.embed_documents(["claim"]) == [[1, 0]]
    assert claims.embed_query("claim") == [1, 0]
    assert encodes == ["document", "query", "symmetric", "symmetric"]
    assert loads == [("shared", {"local_files_only": True})]
