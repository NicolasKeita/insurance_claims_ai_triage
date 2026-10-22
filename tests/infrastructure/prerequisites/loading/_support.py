"""Count model construction without importing or loading any inference framework."""

import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def model_loads(monkeypatch):
    loads = []

    class Model:
        def __init__(self, name, **options):
            loads.append((name, options))

        def get_embedding_dimension(self):
            return 2

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    return loads
