"""Read and write operations for the derived similar-claims Qdrant index."""

from dataclasses import dataclass
from datetime import date
from uuid import UUID

import numpy as np
from qdrant_client import models

from retrieval.models import (
    ClaimRetrievalDocument,
    RetrievalHit,
    RetrievalUnavailableError,
)
from retrieval.representation import CLAIM_RETRIEVAL_REPRESENTATION_VERSION


@dataclass(frozen=True)
class RetrievalPage:
    hits: tuple[RetrievalHit, ...]
    exhausted: bool


class QdrantClaimOperations:
    """Upsert, query and count points of one derived index."""

    def _vectors(self, vectors, count):
        values = np.asarray(vectors, dtype=np.float32)
        if (
            values.shape != (count, self.embeddings.dimension)
            or not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-4)
        ):
            raise ValueError("Expected finite, L2-normalized embeddings with the configured dimension")
        return values.tolist()

    def upsert(self, documents: list[ClaimRetrievalDocument], vectors: list[list[float]]):
        if not documents:
            return
        vectors = self._vectors(vectors, len(documents))
        if any(d.representation_version != CLAIM_RETRIEVAL_REPRESENTATION_VERSION for d in documents):
            raise ValueError("Document representation version does not match this index")
        points = [
            models.PointStruct(
                id=str(document.point_id),
                vector=vector,
                payload={
                    "claim_id": document.claim_id,
                    "claim_type": document.claim_type,
                    "incident_date": document.incident_date.isoformat(),
                    "incident_date_ordinal": document.incident_date.toordinal(),
                    "representation_version": document.representation_version,
                },
            )
            for document, vector in zip(documents, vectors, strict=True)
        ]
        with self._operation():
            self.client.upsert(self.collection, points=points, wait=True)

    def require_collection(self):
        """Read-only; never create/rebuild an index during retrieval."""
        with self._operation():
            if not self.client.collection_exists(self.collection):
                raise RetrievalUnavailableError("Similar claims index is missing; run indexing first")
            self._validate()

    def query(
        self, vector: list[float], before: date, claim_type: str | None, limit: int, offset: int = 0
    ) -> RetrievalPage:
        """
        Foreign/non-UUID points cannot reference our PostgreSQL claims.
        """
        vector = self._vectors([vector], 1)[0]
        conditions = [
            models.FieldCondition(key="incident_date_ordinal", range=models.Range(lt=before.toordinal())),
            models.FieldCondition(
                key="representation_version", match=models.MatchValue(value=CLAIM_RETRIEVAL_REPRESENTATION_VERSION)
            ),
        ]
        if claim_type is not None:
            conditions.append(models.FieldCondition(key="claim_type", match=models.MatchValue(value=claim_type)))
        with self._operation():
            points = self.client.query_points(
                self.collection,
                query=vector,
                query_filter=models.Filter(must=conditions),
                limit=limit,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            ).points
        hits = []
        for point in points:
            try:
                point_id = UUID(str(point.id))
            except ValueError:
                continue
            hits.append(RetrievalHit(point_id, point.score))
        return RetrievalPage(tuple(hits), len(points) < limit)

    def count(self) -> int:
        with self._operation():
            return self.client.count(self.collection, exact=True).count

    def close(self):
        self.client.close()
