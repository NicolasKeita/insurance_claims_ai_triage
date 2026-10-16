"""Normalized lexical hashes for retrieval mechanics, not semantic quality."""

import re
from hashlib import sha256

import numpy as np


class FakeEmbeddings:
    """Normalized lexical hashes for mechanics, never semantic quality evidence."""

    dimension = 128
    model_name = "fake-lexical-embeddings-e2e-v1"
    normalized = True

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for token in re.findall(r"[a-z0-9]+", text.casefold()):
                index = int.from_bytes(sha256(token.encode()).digest()[:4], "big") % self.dimension
                vector[index] += 1
            if not vector.any():
                vector[0] = 1
            vectors.append((vector / np.linalg.norm(vector)).tolist())
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]