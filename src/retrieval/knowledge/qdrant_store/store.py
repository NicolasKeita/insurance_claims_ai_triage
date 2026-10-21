"""Collection lifecycle and checked embedding/chunking contracts."""

from contextlib import contextmanager

from qdrant_client import QdrantClient, models

from retrieval.embeddings import EmbeddingProvider
from retrieval.knowledge.config import KNOWLEDGE_CHUNKING_VERSION, KNOWLEDGE_COLLECTION_VERSION, KnowledgeConfig
from retrieval.knowledge.models import KnowledgeSourceType
from retrieval.knowledge.qdrant_store.filters import PAYLOAD_INDEXES
from retrieval.knowledge.qdrant_store.operations import QdrantKnowledgeOperations
from retrieval.models import IndexCompatibilityError, RetrievalUnavailableError


class QdrantKnowledgeStore(QdrantKnowledgeOperations):
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
            "normalized": True,
            "distance": "COSINE",
            "chunking_version": KNOWLEDGE_CHUNKING_VERSION,
            "max_chunk_chars": self.config.max_chunk_chars,
            "overlap_chars": 0,
        }

    def _validate(self, source_type):
        collection = self.config.collection(source_type)
        info = self.client.get_collection(collection)
        vectors, contract = info.config.params.vectors, self.contract(source_type)
        if (
            not isinstance(vectors, models.VectorParams)
            or vectors.size != contract["embedding_dimension"]
            or vectors.distance != models.Distance.COSINE
            or (info.config.metadata or {}).get("knowledge_rag") != contract
        ):
            raise IndexCompatibilityError(
                f"Collection {collection} has an incompatible knowledge contract; "
                "use a new version or explicitly --recreate the RAG collections"
            )

    def ensure_collection(self, source_type: KnowledgeSourceType, *, recreate: bool = False):
        """
        Only delete an owned RAG collection. Prevent a misconfigured name from destroying another
        application's collection.
        """
        with self._operation():
            collection, contract = self.config.collection(source_type), self.contract(source_type)
            exists = self.client.collection_exists(collection)
            if recreate and exists:
                info = self.client.get_collection(collection)
                existing = (info.config.metadata or {}).get("knowledge_rag")
                if not isinstance(existing, dict) or existing.get("source_type") != source_type.value:
                    raise IndexCompatibilityError("Refusing to recreate a collection not owned by this RAG source type")
                self.client.delete_collection(collection)
                exists = False
            if not exists:
                self.client.create_collection(
                    collection,
                    vectors_config=models.VectorParams(
                        size=contract["embedding_dimension"],
                        distance=models.Distance.COSINE,
                    ),
                    metadata={"knowledge_rag": contract},
                )
            self._validate(source_type)
            info = self.client.get_collection(collection)
            local = (
                self.client.init_options.get("location") == ":memory:"
                or self.client.init_options.get("path") is not None
            )
            if not local:
                for field, schema in PAYLOAD_INDEXES.items():
                    if field not in info.payload_schema:
                        self.client.create_payload_index(collection, field, schema, wait=True)

    def require_collection(self, source_type):
        with self._operation():
            if not self.client.collection_exists(self.config.collection(source_type)):
                raise RetrievalUnavailableError("Knowledge index is missing; run index_knowledge_base.py first")
            self._validate(source_type)

    def count(self, source_type):
        with self._operation():
            return self.client.count(self.config.collection(source_type), exact=True).count

    def close(self):
        self.client.close()
