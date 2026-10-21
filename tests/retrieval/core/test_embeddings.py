"""Test embeddings."""

import subprocess
import sys
from types import SimpleNamespace

from retrieval.embeddings import SentenceTransformerEmbeddingProvider
from tests.retrieval.core._support import components as components


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
