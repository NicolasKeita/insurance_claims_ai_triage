"""No Sentence Transformers import or model I/O until first explicit use."""

from threading import Lock
from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    model_name: str
    normalized: bool

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbeddingProvider:
    normalized = True

    def __init__(self, model_name: str, *, asymmetric: bool = False):
        self.model_name = model_name
        # Existing claim-to-claim indexes retain their symmetric encode contract.
        self.asymmetric = asymmetric
        self._model = None
        self._lock = Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            return self._model

    @property
    def dimension(self) -> int:
        dimension = self._load().get_embedding_dimension()
        if dimension is None or dimension < 1:
            raise ValueError("Embedding model has no sentence embedding dimension")
        return int(dimension)

    def _encode(self, texts: list[str], method: str) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        encoder = getattr(model, method, None) if self.asymmetric else None
        vectors = (encoder if callable(encoder) else model.encode)(
            texts, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, "encode_document")

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], "encode_query")[0]
