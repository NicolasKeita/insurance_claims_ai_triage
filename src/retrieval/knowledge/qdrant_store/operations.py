"""Vector validation, upsert and query for one knowledge collection."""

import numpy as np
from qdrant_client import models

from retrieval.knowledge.config import KNOWLEDGE_CHUNKING_VERSION
from retrieval.knowledge.models import KnowledgeChunk, KnowledgeFilter
from retrieval.knowledge.qdrant_store.filters import knowledge_filter


class QdrantKnowledgeOperations:
    """Read and write points of a knowledge collection."""

    def _vectors(self, vectors, count):
        values = np.asarray(vectors, dtype=np.float32)
        if (
            values.shape != (count, self.embeddings.dimension)
            or not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4)
        ):
            raise ValueError("Expected finite normalized knowledge vectors with configured dimension")
        return values.tolist()

    def upsert(self, source_type, chunks: list[KnowledgeChunk], vectors):
        if not chunks:
            return
        vectors = self._vectors(vectors, len(chunks))
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if chunk.source_type != source_type or chunk.chunking_version != KNOWLEDGE_CHUNKING_VERSION:
                raise ValueError("Chunk belongs to another source type or chunking strategy")
            payload = chunk.model_dump(mode="json")
            payload.update(
                {
                    "effective_from_ordinal": chunk.effective_from.toordinal() if chunk.effective_from else None,
                    "effective_to_ordinal": chunk.effective_to.toordinal() if chunk.effective_to else None,
                }
            )
            points.append(models.PointStruct(id=chunk.chunk_id, vector=vector, payload=payload))
        with self._operation():
            self.client.upsert(self.config.collection(source_type), points=points, wait=True)

    def query(self, source_type, vector, *, filters=None, limit=5, offset=0):
        """
        Only IDs/ranking cross the boundary; text and provenance are re-resolved.
        """
        active = filters or KnowledgeFilter()
        vector = self._vectors([vector], 1)[0]
        with self._operation():
            points = self.client.query_points(
                self.config.collection(source_type),
                query=vector,
                query_filter=knowledge_filter(
                    source_type,
                    product=active.product,
                    language=active.language,
                    source_id=active.source_id,
                    as_of_date=active.as_of_date,
                ),
                limit=limit,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            ).points
        return [(str(p.id), float(p.score)) for p in points if (p.payload or {}).get("chunk_id") == str(p.id)], len(
            points
        ) < limit
