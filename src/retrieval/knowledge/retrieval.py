"""Authoritative re-resolution of retrieved knowledge chunks."""

import logging

from retrieval.config import validate_limit
from retrieval.knowledge.models import KnowledgeFilter, KnowledgeSourceType, RetrievedKnowledgeChunk, applicable_on
from retrieval.knowledge.qdrant_store import QdrantKnowledgeStore
from retrieval.knowledge.sources import KnowledgeCatalog
from retrieval.models import RetrievalUnavailableError

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    def __init__(self, store: QdrantKnowledgeStore):
        self.store = store

    def search(
        self,
        query: str,
        source_type: KnowledgeSourceType,
        *,
        filters: KnowledgeFilter | None = None,
        limit: int = 5,
    ) -> list[RetrievedKnowledgeChunk]:
        """
        Reload local authoritative sources: changed/deleted chunks are ignored.

        INFO contains operational IDs only. Queries may contain sensitive data.
        """
        active = filters or KnowledgeFilter()
        validate_limit(limit)
        if not query.strip() or len(query) > 4000:
            raise ValueError("query must contain 1 to 4000 characters")
        source_type = KnowledgeSourceType(source_type)
        self.store.require_collection(source_type)
        catalog, vector = self._prepare(query)
        results = self._collect(source_type, vector, catalog, active, limit)
        self._log(query, source_type, active, results)
        return results

    def _prepare(self, query):
        try:
            catalog = KnowledgeCatalog.load(self.store.config.knowledge_dir, self.store.config.max_chunk_chars)
            vector = self.store.embeddings.embed_query(query)
        except Exception as error:
            raise RetrievalUnavailableError("Knowledge sources or embedding model are unavailable") from error
        return catalog, vector

    def _collect(self, source_type, vector, catalog, active, limit):
        results, seen = [], set()
        page_size = max(20, limit * 2)
        for offset in range(0, 1000, page_size):
            hits, exhausted = self.store.query(
                source_type, vector, filters=active, limit=min(page_size, 1000 - offset), offset=offset
            )
            for chunk_id, score in hits:
                chunk = catalog.chunks.get(chunk_id)
                if not self._acceptable(chunk, chunk_id, seen, source_type, active):
                    continue
                seen.add(chunk_id)
                results.append(RetrievedKnowledgeChunk(**chunk.model_dump(), similarity_score=score))
                if len(results) == limit:
                    break
            if len(results) == limit or exhausted:
                break
        return results

    @staticmethod
    def _acceptable(chunk, chunk_id, seen, source_type, active):
        """Re-resolve authority and reapply every filter to the local catalog chunk."""
        if chunk is None or chunk_id in seen or chunk.source_type != source_type:
            return False
        if active.product is not None and chunk.product != active.product:
            return False
        if active.language is not None and chunk.language != active.language:
            return False
        if active.source_id is not None and chunk.source_id != active.source_id:
            return False
        return applicable_on(chunk, active.as_of_date)

    def _log(self, query, source_type, active, results):
        logger.info(
            "knowledge_retrieval type=%s filters=%s results=%s",
            source_type.value,
            {
                "product": active.product,
                "language": active.language,
                "source_id": active.source_id,
                "as_of_date": str(active.as_of_date),
            },
            [(c.chunk_id, c.similarity_score) for c in results],
        )
        logger.debug("knowledge_retrieval query=%r", query)

    def close(self):
        self.store.close()
