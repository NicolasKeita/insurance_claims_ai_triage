"""Batch indexer that upserts derived retrieval documents."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import UUID

from retrieval.embeddings import EmbeddingProvider
from retrieval.models import StoredRetrievalClaim
from retrieval.qdrant_store import QdrantClaimStore
from retrieval.representation import build_retrieval_document


class RetrievalRepository(Protocol):
    def get(self, claim_id: str) -> StoredRetrievalClaim: ...
    def get_many(self, point_ids: list[UUID]) -> dict[UUID, StoredRetrievalClaim]: ...
    def iter_batches(self, batch_size: int): ...


@dataclass(frozen=True)
class IndexingStats:
    claims_read: int
    claims_embedded: int
    points_upserted: int
    collection: str


class SimilarClaimsIndexer:
    def __init__(
        self,
        repository: RetrievalRepository,
        embeddings: EmbeddingProvider,
        store: QdrantClaimStore,
        batch_size: int = 32,
    ):
        if not 1 <= batch_size <= 1024:
            raise ValueError("batch_size must be between 1 and 1024")
        self.repository, self.embeddings, self.store = repository, embeddings, store
        self.batch_size = batch_size

    def _upsert(self, claims):
        documents = [build_retrieval_document(claim) for claim in claims]
        vectors = self.embeddings.embed_documents([d.canonical_text for d in documents])
        self.store.upsert(documents, vectors)

    def index_all(self, *, recreate: bool = False) -> IndexingStats:
        self.store.ensure_collection(recreate=recreate)
        count = 0
        for claims in self.repository.iter_batches(self.batch_size):
            self._upsert(claims)
            count += len(claims)
        return IndexingStats(count, count, count, self.store.collection)

    def upsert_claim(self, claim_id: str) -> IndexingStats:
        claim = self.repository.get(claim_id)
        self.store.ensure_collection()
        self._upsert([claim])
        return IndexingStats(1, 1, 1, self.store.collection)

    def write_manifest(self, path: Path, stats: IndexingStats) -> dict:
        manifest = {
            "collection_name": self.store.collection,
            **self.store.contract(),
            "indexed_claim_count": self.store.count(),
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "last_run": asdict(stats),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return manifest
