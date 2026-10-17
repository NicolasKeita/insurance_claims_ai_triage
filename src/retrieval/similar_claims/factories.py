"""Factories wiring the real Qdrant store and PostgreSQL retrieval repository."""

import os

from qdrant_client import QdrantClient

from retrieval.config import RetrievalConfig
from retrieval.embeddings import SentenceTransformerEmbeddingProvider
from retrieval.qdrant_store import QdrantClaimStore
from retrieval.similar_claims.service import SimilarClaimsService


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
