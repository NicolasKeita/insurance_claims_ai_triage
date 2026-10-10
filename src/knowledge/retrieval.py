import logging
from datetime import date

from knowledge.models import KnowledgeSourceType, RetrievedKnowledgeChunk, applicable_on
from knowledge.qdrant_store import QdrantKnowledgeStore
from knowledge.sources import KnowledgeCatalog
from retrieval.config import validate_limit
from retrieval.models import RetrievalUnavailableError

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    def __init__(self, store: QdrantKnowledgeStore):
        self.store = store

    def search(self, query: str, source_type: KnowledgeSourceType, *, product: str | None = None,
               language: str | None = None, source_id: str | None = None, as_of_date: date | None = None,
               limit: int = 5) -> list[RetrievedKnowledgeChunk]:
        validate_limit(limit)
        if not query.strip() or len(query) > 4000:
            raise ValueError("query must contain 1 to 4000 characters")
        source_type = KnowledgeSourceType(source_type)
        self.store.require_collection(source_type)
        try:
            # Reload local authoritative sources: changed/deleted chunks are ignored.
            catalog = KnowledgeCatalog.load(self.store.config.knowledge_dir, self.store.config.max_chunk_chars)
            vector = self.store.embeddings.embed_query(query)
        except Exception as error:
            raise RetrievalUnavailableError("Knowledge sources or embedding model are unavailable") from error
        results, seen = [], set()
        page_size = max(20, limit * 2)
        for offset in range(0, 1000, page_size):
            hits, exhausted = self.store.query(
                source_type, vector, product=product, language=language, source_id=source_id, as_of_date=as_of_date,
                limit=min(page_size, 1000 - offset), offset=offset,
            )
            for chunk_id, score in hits:
                chunk = catalog.chunks.get(chunk_id)
                if (chunk is None or chunk_id in seen or chunk.source_type != source_type
                        or (product is not None and chunk.product != product)
                        or (language is not None and chunk.language != language)
                        or (source_id is not None and chunk.source_id != source_id)
                        or not applicable_on(chunk, as_of_date)):
                    continue
                seen.add(chunk_id)
                results.append(RetrievedKnowledgeChunk(**chunk.model_dump(), similarity_score=score))
                if len(results) == limit:
                    break
            if len(results) == limit or exhausted:
                break
        # INFO contains operational IDs only. Queries may contain sensitive data.
        logger.info("knowledge_retrieval type=%s filters=%s results=%s", source_type.value,
                    {"product": product, "language": language, "source_id": source_id, "as_of_date": str(as_of_date)},
                    [(c.chunk_id, c.similarity_score) for c in results])
        logger.debug("knowledge_retrieval query=%r", query)
        return results

    def close(self):
        self.store.close()
