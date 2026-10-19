"""Derived index with a persisted embedding contract and strict filters."""

from contextlib import contextmanager

from qdrant_client import QdrantClient, models

from retrieval.embeddings import EmbeddingProvider
from retrieval.models import IndexCompatibilityError, RetrievalUnavailableError
from retrieval.qdrant_operations import QdrantClaimOperations, RetrievalPage
from retrieval.representation import CLAIM_RETRIEVAL_REPRESENTATION_VERSION

__all__ = ["QdrantClaimStore", "RetrievalPage"]


class QdrantClaimStore(QdrantClaimOperations):
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
        if (
            not isinstance(vectors, models.VectorParams)
            or vectors.size != contract["embedding_dimension"]
            or vectors.distance != models.Distance.COSINE
            or (info.config.metadata or {}).get("claim_retrieval") != contract
        ):
            raise IndexCompatibilityError(
                f"Collection {self.collection} has an incompatible embedding contract; "
                "use a new collection/version and reindex, or explicitly --recreate the derived index"
            )

    def ensure_collection(self, *, recreate: bool = False):
        """
        Resolve contract before any explicitly requested destructive action.

        Create indexes only once (Qdrant local mode has no payload indexes).
        """
        with self._operation():
            contract = self.contract()
            exists = self.client.collection_exists(self.collection)
            if recreate and exists:
                self.client.delete_collection(self.collection)
                exists = False
            if not exists:
                self.client.create_collection(
                    self.collection,
                    vectors_config=models.VectorParams(
                        size=contract["embedding_dimension"],
                        distance=models.Distance.COSINE,
                    ),
                    metadata={"claim_retrieval": contract},
                )
            self._validate()
            info = self.client.get_collection(self.collection)
            local = (
                self.client.init_options.get("location") == ":memory:"
                or self.client.init_options.get("path") is not None
            )
            if not local:
                for field, schema in (
                    ("incident_date_ordinal", models.PayloadSchemaType.INTEGER),
                    ("claim_type", models.PayloadSchemaType.KEYWORD),
                    ("representation_version", models.PayloadSchemaType.KEYWORD),
                ):
                    if field not in info.payload_schema:
                        self.client.create_payload_index(self.collection, field, schema, wait=True)
