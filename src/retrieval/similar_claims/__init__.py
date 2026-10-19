"""Batch indexer, PostgreSQL-authoritative search and their factories."""

from retrieval.similar_claims.factories import create_retrieval_components, create_similar_claims_service
from retrieval.similar_claims.indexer import IndexingStats, RetrievalRepository, SimilarClaimsIndexer
from retrieval.similar_claims.service import SimilarClaimsService

__all__ = [
    "IndexingStats",
    "RetrievalRepository",
    "SimilarClaimsIndexer",
    "SimilarClaimsService",
    "create_retrieval_components",
    "create_similar_claims_service",
]
