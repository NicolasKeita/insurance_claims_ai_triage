"""PostgreSQL-authoritative similar-claims search."""

from retrieval.config import DEFAULT_LIMIT, validate_limit
from retrieval.embeddings import EmbeddingProvider
from retrieval.models import RetrievalUnavailableError, SimilarClaim, SimilarClaimsResponse
from retrieval.qdrant_store import QdrantClaimStore
from retrieval.representation import build_retrieval_document
from retrieval.similar_claims.indexer import RetrievalRepository

MAX_CANDIDATES = 1000


class SimilarClaimsService:
    def __init__(
        self,
        repository: RetrievalRepository,
        embeddings: EmbeddingProvider,
        store: QdrantClaimStore,
        *,
        same_claim_type: bool = True,
    ):
        self.repository, self.embeddings, self.store = repository, embeddings, store
        self.same_claim_type = same_claim_type

    def search(self, current_claim_id: str, limit: int = DEFAULT_LIMIT) -> SimilarClaimsResponse:
        """
        Unknown always yields 404 before model I/O.
        """
        validate_limit(limit)
        current = self.repository.get(current_claim_id)
        self.store.require_collection()
        document = build_retrieval_document(current)
        vector = self._query_vector(document)
        results = self._search_pages(current, vector, limit)
        return SimilarClaimsResponse(
            claim_id=current.claim_id,
            representation_version=document.representation_version,
            results=results,
        )

    def _query_vector(self, document):
        try:
            return self.embeddings.embed_query(document.canonical_text)
        except Exception as error:
            raise RetrievalUnavailableError("Similar claims embedding model is unavailable") from error

    def _search_pages(self, current, vector, limit):
        results = []
        seen = {current.point_id}
        page_size = max(20, limit * 4)
        for offset in range(0, MAX_CANDIDATES, page_size):
            size = min(page_size, MAX_CANDIDATES - offset)
            page = self.store.query(
                vector, current.incident_date, current.facts.claim_type if self.same_claim_type else None, size, offset
            )
            self._extend(results, seen, current, page, limit)
            if len(results) == limit or page.exhausted:
                break
        return results

    def _extend(self, results, seen, current, page, limit):
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
            results.append(_similar_claim(candidate, hit))
            if len(results) == limit:
                break

    def close(self):
        self.store.close()


def _similar_claim(candidate, hit):
    return SimilarClaim(
        claim_id=candidate.claim_id,
        similarity_score=hit.similarity_score,
        incident_date=candidate.incident_date,
        claim_type=candidate.facts.claim_type,
        collision_type=candidate.facts.collision_type,
        repair_amount=candidate.facts.repair_amount,
        repair_currency=candidate.facts.repair_currency,
    )
