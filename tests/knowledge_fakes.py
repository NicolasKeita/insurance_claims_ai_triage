import json
from hashlib import sha256

import numpy as np


class FakeEmbeddingProvider:
    """Deterministic vectors for mechanics tests, never a semantic benchmark."""
    dimension = 8
    model_name = "fake-knowledge-embeddings-v1"
    normalized = True

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            values = np.array(list(sha256(text.encode()).digest()[:self.dimension]), dtype=np.float32) + 1
            vectors.append((values / np.linalg.norm(values)).tolist())
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class FakeStructuredLlm:
    def __init__(self, output=None):
        self.output = output
        self.prompts = []

    def generate(self, *, prompt, response_model):
        self.prompts.append(prompt)
        data = json.loads(prompt.split("INPUT DATA:\n", 1)[1])
        output = self.output if self.output is not None else {
            "answer": "Demo answer from retrieved evidence.",
            "cited_chunk_ids": [data["retrieved_sources"][0]["chunk_id"]],
            "insufficient_evidence": False,
        }
        return response_model.model_validate(output)
