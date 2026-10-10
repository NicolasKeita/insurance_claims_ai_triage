"""Derived index with a persisted embedding contract and strict filters."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from uuid import UUID

import numpy as np
from qdrant_client import QdrantClient, models

from retrieval.embeddings import EmbeddingProvider
from retrieval.models import (
    ClaimRetrievalDocument, IndexCompatibilityError, RetrievalHit, RetrievalUnavailableError,
)
from retrieval.representation import CLAIM_RETRIEVAL_REPRESENTATION_VERSION


@dataclass(frozen=True)
class RetrievalPage:
    hits: tuple[RetrievalHit, ...]
    exhausted: bool


class QdrantClaimStore:
    def __init__(self, client: QdrantClient, collection: str, embeddings: EmbeddingProvider):
        self.client = client
        self.collection = collection
        self.embeddings = embeddings

    @contextmanager
    def _operation(self):
        try:
            yield
        except RetrievalUnavailableError:
            raise
        except Exception as error:
            raise RetrievalUnavailableError("Similar claims index is unavailable") from error

    def contract(self) -> dict:
        if not self.embeddings.normalized:
            raise IndexCompatibilityError("Similar claims requires normalized embeddings")
        return {
            "representation_version": CLAIM_RETRIEVAL_REPRESENTATION_VERSION,
            "embedding_model": self.embeddings.model_name,
            "embedding_dimension": self.embeddings.dimension,
            "normalized": True,
            "distance": "COSINE",
        }

    def _validate(self):
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        contract = self.contract()
        if (not isinstance(vectors, models.VectorParams)
                or vectors.size != contract["embedding_dimension"]
                or vectors.distance != models.Distance.COSINE
                or (info.config.metadata or {}).get("claim_retrieval") != contract):
            raise IndexCompatibilityError(
                f"Collection {self.collection} has an incompatible embedding contract; "
                "use a new collection/version and reindex, or explicitly --recreate the derived index"
            )

    def ensure_collection(self, *, recreate: bool = False):
        with self._operation():
            # Resolve contract before any explicitly requested destructive action.
            contract = self.contract()
            exists = self.client.collection_exists(self.collection)
            if recreate and exists:
                self.client.delete_collection(self.collection)
                exists = False
            if not exists:
                self.client.create_collection(
                    self.collection,
                    vectors_config=models.VectorParams(
                        size=contract["embedding_dimension"], distance=models.Distance.COSINE,
                    ),
                    metadata={"claim_retrieval": contract},
                )
            self._validate()
            # Create indexes only once (Qdrant local mode has no payload indexes).
            info = self.client.get_collection(self.collection)
            local = (self.client.init_options.get("location") == ":memory:"
                     or self.client.init_options.get("path") is not None)
            if not local:
                for field, schema in (
                    ("incident_date_ordinal", models.PayloadSchemaType.INTEGER),
                    ("claim_type", models.PayloadSchemaType.KEYWORD),
                    ("representation_version", models.PayloadSchemaType.KEYWORD),
                ):
                    if field not in info.payload_schema:
                        self.client.create_payload_index(self.collection, field, schema, wait=True)

    def require_collection(self):
        """Read-only; never create/rebuild an index during retrieval."""
        with self._operation():
            if not self.client.collection_exists(self.collection):
                raise RetrievalUnavailableError("Similar claims index is missing; run indexing first")
            self._validate()

    def _vectors(self, vectors, count):
        values = np.asarray(vectors, dtype=np.float32)
        if (values.shape != (count, self.embeddings.dimension)
                or not np.isfinite(values).all()
                or not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-4)):
            raise ValueError("Expected finite, L2-normalized embeddings with the configured dimension")
        return values.tolist()

    def upsert(self, documents: list[ClaimRetrievalDocument], vectors: list[list[float]]):
        if not documents:
            return
        vectors = self._vectors(vectors, len(documents))
        if any(d.representation_version != CLAIM_RETRIEVAL_REPRESENTATION_VERSION for d in documents):
            raise ValueError("Document representation version does not match this index")
        points = [models.PointStruct(
            id=str(document.point_id), vector=vector,
            payload={
                "claim_id": document.claim_id, "claim_type": document.claim_type,
                "incident_date": document.incident_date.isoformat(),
                "incident_date_ordinal": document.incident_date.toordinal(),
                "representation_version": document.representation_version,
            },
        ) for document, vector in zip(documents, vectors, strict=True)]
        with self._operation():
            self.client.upsert(self.collection, points=points, wait=True)

    def query(self, vector: list[float], before: date, claim_type: str | None,
              limit: int, offset: int = 0) -> RetrievalPage:
        vector = self._vectors([vector], 1)[0]
        conditions = [
            models.FieldCondition(key="incident_date_ordinal", range=models.Range(lt=before.toordinal())),
            models.FieldCondition(key="representation_version", match=models.MatchValue(
                value=CLAIM_RETRIEVAL_REPRESENTATION_VERSION)),
        ]
        if claim_type is not None:
            conditions.append(models.FieldCondition(key="claim_type", match=models.MatchValue(value=claim_type)))
        with self._operation():
            points = self.client.query_points(
                self.collection, query=vector, query_filter=models.Filter(must=conditions),
                limit=limit, offset=offset, with_payload=False, with_vectors=False,
            ).points
        hits = []
        for point in points:
            try:
                point_id = UUID(str(point.id))
            except ValueError:
                continue  # Foreign/non-UUID points cannot reference our PostgreSQL claims.
            hits.append(RetrievalHit(point_id, point.score))
        return RetrievalPage(tuple(hits), len(points) < limit)

    def count(self) -> int:
        with self._operation():
            return self.client.count(self.collection, exact=True).count

    def close(self):
        self.client.close()
