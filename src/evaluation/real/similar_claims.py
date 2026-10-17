"""Similar-claims search over the real repository and Qdrant store."""

import os


class SimilarClaimsSteps:
    """Search similar claims without a fake repository or fake vectors."""

    def similar(self, run):
        """Search similar claims through the real repository and Qdrant store."""
        observed = run.observed
        if not self.config.include_similar:
            self._skip(observed, "similar_claims", "Disabled by configuration")
            return
        if not self._available(
            observed, "similar_claims", ["postgresql", "qdrant", "similar_claims_collection", "embeddings"]
        ):
            return
        if observed.components.get("history") and observed.components["history"].status != "PASS":
            self._skip(observed, "similar_claims", "Controlled claim has no matching stored source")
            return
        from persistence.retrieval import RetrievalClaimRepository
        from retrieval.qdrant_store import QdrantClaimStore
        from retrieval.similar_claims import SimilarClaimsIndexer, SimilarClaimsService

        config = self.resources["retrieval_config"]
        embeddings = self.similar_embeddings(config)
        repository = RetrievalClaimRepository(self.resources["session_factory"])
        store = QdrantClaimStore(self.resources["qdrant_client"], config.collection, embeddings)
        service = SimilarClaimsService(repository, embeddings, store, same_claim_type=config.same_claim_type)
        if self.config.isolated_demo_stack:
            self._record(
                observed,
                "similar_indexing",
                lambda: SimilarClaimsIndexer(repository, embeddings, store).index_all(),
                inputs={"collection": config.collection, "scope": "Controlled claims in this run's isolated schema"},
            )
        result = self._record(
            observed,
            "similar_claims",
            lambda: service.search(run.case.claim.claim_id),
            inputs={"claim_id": run.case.claim.claim_id},
        )
        if result is not None:
            observed.facts["similar_claims"] = result.model_dump(mode="json")

    def similar_embeddings(self, config):
        """Reuse the locally loadable embedding provider across scenarios."""
        from retrieval.embeddings import SentenceTransformerEmbeddingProvider

        embeddings = self.resources.get("similar_embeddings")
        if embeddings is None:
            embeddings = SentenceTransformerEmbeddingProvider(config.embedding_model, local_files_only=True)
            self.resources["similar_embeddings"] = embeddings
        return embeddings

    def similar_service(self):
        """Load the real similar-claims service, or fail the agent tool explicitly."""
        from retrieval.models import RetrievalUnavailableError

        available = ("qdrant", "similar_claims_collection", "embeddings")
        if not self.config.include_similar or not all(self.preflight.available(name) for name in available):
            raise RetrievalUnavailableError("Similar claims unavailable or disabled by evaluation configuration")
        from qdrant_client import QdrantClient

        from persistence.retrieval import RetrievalClaimRepository
        from retrieval.qdrant_store import QdrantClaimStore
        from retrieval.similar_claims import SimilarClaimsService

        config = self.resources["retrieval_config"]
        embeddings = self.similar_embeddings(config)
        client = QdrantClient(
            url=config.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=self.config.timeout_seconds
        )
        return SimilarClaimsService(
            RetrievalClaimRepository(self.resources["session_factory"]),
            embeddings,
            QdrantClaimStore(client, config.collection, embeddings),
            same_claim_type=config.same_claim_type,
        )
