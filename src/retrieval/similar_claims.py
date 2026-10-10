"""Batch indexer and PostgreSQL-authoritative search orchestration."""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import UUID

from qdrant_client import QdrantClient

from retrieval.config import DEFAULT_LIMIT, RetrievalConfig, validate_limit
from retrieval.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from retrieval.models import (
    RetrievalUnavailableError, SimilarClaim, SimilarClaimsResponse, StoredRetrievalClaim,
)
from retrieval.qdrant_store import QdrantClaimStore
from retrieval.representation import build_retrieval_document

MAX_CANDIDATES = 1000


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
    def __init__(self, repository: RetrievalRepository, embeddings: EmbeddingProvider,
                 store: QdrantClaimStore, batch_size: int = 32):
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
            "collection_name": self.store.collection, **self.store.contract(),
            "indexed_claim_count": self.store.count(),
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "last_run": asdict(stats),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return manifest


class SimilarClaimsService:
    def __init__(self, repository: RetrievalRepository, embeddings: EmbeddingProvider,
                 store: QdrantClaimStore, *, same_claim_type: bool = True):
        self.repository, self.embeddings, self.store = repository, embeddings, store
        self.same_claim_type = same_claim_type

    def search(self, current_claim_id: str, limit: int = DEFAULT_LIMIT) -> SimilarClaimsResponse:
        validate_limit(limit)
        current = self.repository.get(current_claim_id)  # Unknown always yields 404 before model I/O.
        self.store.require_collection()
        document = build_retrieval_document(current)
        try:
            vector = self.embeddings.embed_query(document.canonical_text)
        except Exception as error:
            raise RetrievalUnavailableError("Similar claims embedding model is unavailable") from error
        results = []
        seen = {current.point_id}
        page_size = max(20, limit * 4)
        for offset in range(0, MAX_CANDIDATES, page_size):
            size = min(page_size, MAX_CANDIDATES - offset)
            page = self.store.query(vector, current.incident_date,
                                    current.facts.claim_type if self.same_claim_type else None,
                                    size, offset)
            authoritative = self.repository.get_many([hit.point_id for hit in page.hits])
            for hit in page.hits:
                candidate = authoritative.get(hit.point_id)
                if candidate is None or hit.point_id in seen:
                    continue
                seen.add(hit.point_id)
                if candidate.incident_date >= current.incident_date:
                    continue
                if self.same_claim_type and candidate.facts.claim_type != current.facts.claim_type:
                    continue
                results.append(SimilarClaim(
                    claim_id=candidate.claim_id, similarity_score=hit.similarity_score,
                    incident_date=candidate.incident_date, claim_type=candidate.facts.claim_type,
                    collision_type=candidate.facts.collision_type,
                    repair_amount=candidate.facts.repair_amount,
                    repair_currency=candidate.facts.repair_currency,
                ))
                if len(results) == limit:
                    break
            if len(results) == limit or page.exhausted:
                break
        return SimilarClaimsResponse(
            claim_id=current.claim_id, representation_version=document.representation_version,
            results=results,
        )

    def close(self):
        self.store.close()


def create_retrieval_components(factory, config: RetrievalConfig | None = None):
    from persistence.retrieval import RetrievalClaimRepository

    config = config or RetrievalConfig.from_env()
    embeddings = SentenceTransformerEmbeddingProvider(config.embedding_model)
    client = QdrantClient(url=config.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=10)
    return RetrievalClaimRepository(factory), embeddings, QdrantClaimStore(client, config.collection, embeddings)


def create_similar_claims_service(factory) -> SimilarClaimsService:
    config = RetrievalConfig.from_env()
    repository, embeddings, store = create_retrieval_components(factory, config)
    return SimilarClaimsService(repository, embeddings, store, same_claim_type=config.same_claim_type)
