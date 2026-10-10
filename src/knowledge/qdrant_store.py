"""Two derived collections with checked embedding/chunking contracts."""

from contextlib import contextmanager
from datetime import date

import numpy as np
from qdrant_client import QdrantClient, models

from knowledge.config import KNOWLEDGE_CHUNKING_VERSION, KNOWLEDGE_COLLECTION_VERSION, KnowledgeConfig
from knowledge.models import KnowledgeChunk, KnowledgeSourceType
from retrieval.embeddings import EmbeddingProvider
from retrieval.models import IndexCompatibilityError, RetrievalUnavailableError

PAYLOAD_INDEXES = {
    "product": models.PayloadSchemaType.KEYWORD,
    "source_type": models.PayloadSchemaType.KEYWORD,
    "language": models.PayloadSchemaType.KEYWORD,
    "source_id": models.PayloadSchemaType.KEYWORD,
    "effective_from_ordinal": models.PayloadSchemaType.INTEGER,
    "effective_to_ordinal": models.PayloadSchemaType.INTEGER,
}


def knowledge_filter(source_type: KnowledgeSourceType, *, product: str | None = None,
                     language: str | None = None, source_id: str | None = None,
                     as_of_date: date | None = None) -> models.Filter:
    conditions = [models.FieldCondition(key="source_type", match=models.MatchValue(value=source_type.value))]
    for field, value in (("product", product), ("language", language), ("source_id", source_id)):
        if value is not None:
            conditions.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
    if as_of_date is not None:
        for field, bounds in (
            ("effective_from_ordinal", models.Range(lte=as_of_date.toordinal())),
            ("effective_to_ordinal", models.Range(gt=as_of_date.toordinal())),
        ):
            conditions.append(models.Filter(should=[
                models.IsNullCondition(is_null=models.PayloadField(key=field)),
                models.FieldCondition(key=field, range=bounds),
            ]))
    return models.Filter(must=conditions)


class QdrantKnowledgeStore:
    def __init__(self, client: QdrantClient, embeddings: EmbeddingProvider, config: KnowledgeConfig):
        self.client, self.embeddings, self.config = client, embeddings, config

    @contextmanager
    def _operation(self):
        try:
            yield
        except RetrievalUnavailableError:
            raise
        except Exception as error:
            raise RetrievalUnavailableError("Knowledge index is unavailable") from error

    def contract(self, source_type: KnowledgeSourceType) -> dict:
        if not self.embeddings.normalized:
            raise IndexCompatibilityError("Knowledge RAG requires normalized embeddings")
        return {
            "collection_version": KNOWLEDGE_COLLECTION_VERSION,
            "source_type": source_type.value,
            "embedding_model": self.embeddings.model_name,
            "embedding_dimension": self.embeddings.dimension,
            "embedding_mode": "query_document",
            "normalized": True, "distance": "COSINE",
            "chunking_version": KNOWLEDGE_CHUNKING_VERSION,
            "max_chunk_chars": self.config.max_chunk_chars, "overlap_chars": 0,
        }

    def _validate(self, source_type):
        collection = self.config.collection(source_type)
        info = self.client.get_collection(collection)
        vectors, contract = info.config.params.vectors, self.contract(source_type)
        if (not isinstance(vectors, models.VectorParams)
                or vectors.size != contract["embedding_dimension"]
                or vectors.distance != models.Distance.COSINE
                or (info.config.metadata or {}).get("knowledge_rag") != contract):
            raise IndexCompatibilityError(
                f"Collection {collection} has an incompatible knowledge contract; "
                "use a new version or explicitly --recreate the RAG collections"
            )

    def ensure_collection(self, source_type: KnowledgeSourceType, *, recreate: bool = False):
        with self._operation():
            collection, contract = self.config.collection(source_type), self.contract(source_type)
            exists = self.client.collection_exists(collection)
            if recreate and exists:
                # Only delete an owned RAG collection. Prevent a misconfigured name
                # from destroying another application's collection.
                info = self.client.get_collection(collection)
                existing = (info.config.metadata or {}).get("knowledge_rag")
                if not isinstance(existing, dict) or existing.get("source_type") != source_type.value:
                    raise IndexCompatibilityError("Refusing to recreate a collection not owned by this RAG source type")
                self.client.delete_collection(collection)
                exists = False
            if not exists:
                self.client.create_collection(collection, vectors_config=models.VectorParams(
                    size=contract["embedding_dimension"], distance=models.Distance.COSINE,
                ), metadata={"knowledge_rag": contract})
            self._validate(source_type)
            info = self.client.get_collection(collection)
            local = self.client.init_options.get("location") == ":memory:" or self.client.init_options.get("path") is not None
            if not local:
                for field, schema in PAYLOAD_INDEXES.items():
                    if field not in info.payload_schema:
                        self.client.create_payload_index(collection, field, schema, wait=True)

    def require_collection(self, source_type):
        with self._operation():
            if not self.client.collection_exists(self.config.collection(source_type)):
                raise RetrievalUnavailableError("Knowledge index is missing; run index_knowledge_base.py first")
            self._validate(source_type)

    def _vectors(self, vectors, count):
        values = np.asarray(vectors, dtype=np.float32)
        if (values.shape != (count, self.embeddings.dimension) or not np.isfinite(values).all()
                or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4)):
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
            payload.update({
                "effective_from_ordinal": chunk.effective_from.toordinal() if chunk.effective_from else None,
                "effective_to_ordinal": chunk.effective_to.toordinal() if chunk.effective_to else None,
            })
            points.append(models.PointStruct(id=chunk.chunk_id, vector=vector, payload=payload))
        with self._operation():
            self.client.upsert(self.config.collection(source_type), points=points, wait=True)

    def query(self, source_type, vector, *, product=None, language=None, source_id=None, as_of_date=None, limit=5, offset=0):
        vector = self._vectors([vector], 1)[0]
        with self._operation():
            points = self.client.query_points(
                self.config.collection(source_type), query=vector,
                query_filter=knowledge_filter(source_type, product=product, language=language, source_id=source_id, as_of_date=as_of_date),
                limit=limit, offset=offset, with_payload=["chunk_id"], with_vectors=False,
            ).points
        # Only IDs/ranking cross the boundary; text and provenance are re-resolved.
        return [(str(p.id), float(p.score)) for p in points
                if (p.payload or {}).get("chunk_id") == str(p.id)], len(points) < limit

    def count(self, source_type):
        with self._operation():
            return self.client.count(self.config.collection(source_type), exact=True).count

    def close(self):
        self.client.close()
