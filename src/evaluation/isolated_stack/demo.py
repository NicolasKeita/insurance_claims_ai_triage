"""Ephemeral real demo stack: one owned schema plus new random collections."""

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from evaluation.isolated_stack.postgres import isolated_postgres

AUDIT_SCOPE = "Explicit isolated demo evaluation; no existing application rows or collections modified"


class IsolatedDemoSetup:
    """Owns the disposable schema, collections and audit document of one run."""

    def __init__(self, config, preflight, output_dir):
        self.config, self.preflight = config, preflight
        self.output_dir = Path(output_dir)
        self.client = preflight.resources["qdrant_client"]
        suffix = uuid4().hex
        self.names = [f"evaluation_e2e_{kind}_{suffix}" for kind in ("policy", "procedure", "claims")]
        self.audit = {"scope": AUDIT_SCOPE, "collections": self.names, "cleaned_up": False}

    def write_audit(self):
        path = self.output_dir / "isolated_setup.json"
        path.write_text(json.dumps(self.audit, indent=2) + "\n", encoding="utf-8")

    def bind_postgres(self, stack):
        original = self.preflight.resources.get("database_engine")
        if original is not None:
            original.dispose()
        self.preflight.resources.update(
            database_engine=stack.engine, session_factory=stack.factory, isolated_schema=stack.schema
        )
        self.audit["schema"] = stack.schema
        knowledge_config = replace(
            self.preflight.resources["knowledge_config"],
            policy_collection=self.names[0],
            procedure_collection=self.names[1],
        )
        retrieval_config = replace(self.preflight.resources["retrieval_config"], collection=self.names[2])
        self.preflight.resources.update(knowledge_config=knowledge_config, retrieval_config=retrieval_config)

    def index(self):
        from knowledge.indexing import KnowledgeIndexer
        from knowledge.qdrant_store import QdrantKnowledgeStore
        from retrieval.embeddings import SentenceTransformerEmbeddingProvider
        from retrieval.qdrant_store import QdrantClaimStore

        if self.config.include_rag:
            store = QdrantKnowledgeStore(
                self.client, self.preflight.resources["embeddings"], self.preflight.resources["knowledge_config"]
            )
            self.audit["knowledge_index"] = KnowledgeIndexer(store).index_all(
                self.output_dir / "isolated_knowledge_index_manifest.json"
            )
        if self.config.include_similar:
            config = self.preflight.resources["retrieval_config"]
            embeddings = SentenceTransformerEmbeddingProvider(config.embedding_model, local_files_only=True)
            self.preflight.resources["similar_embeddings"] = embeddings
            QdrantClaimStore(self.client, self.names[2], embeddings).ensure_collection()

    def update_rows(self):
        for row in self.preflight.rows:
            if row["component"] == "postgresql":
                row.update(schema_available=True, isolated_schema=self.preflight.resources["isolated_schema"])
            elif row["component"] == "knowledge_collections" and self.config.include_rag:
                row.update(
                    status="PASS",
                    reason=None,
                    collections=self.names[:2],
                    setup="New owned isolated demo collections indexed using real embeddings",
                )
            elif row["component"] == "similar_claims_collection" and self.config.include_similar:
                row.update(
                    status="PASS",
                    reason=None,
                    collections=self.names[2:],
                    setup="New owned isolated demo collection; indexed from run's real PostgreSQL rows per scenario",
                )

    def cleanup(self):
        errors = []
        for name in self.names:
            try:
                if self.client.collection_exists(name):
                    self.client.delete_collection(name)
            except Exception as error:
                errors.append(f"{name}: {type(error).__name__}: {error}")
        self.audit.update(
            cleaned_up=not errors,
            collection_cleanup_errors=errors,
            schema_cleanup="Owned schema is dropped by the surrounding PostgreSQL context",
        )


@contextmanager
def isolated_demo_stack(config, preflight, output_dir):
    """Prepare explicitly requested ephemeral real services using existing indexers.

    Only a safe dedicated test database and new randomly named collections are
    mutated. Existing app data and existing collections are never imported or
    overwritten. All created resources are removed on context exit.
    """
    if not config.database_url or not config.qdrant_url:
        raise ValueError("--isolated-demo-stack requires explicit --database-url and --qdrant-url")
    if not preflight.available("postgresql") or not preflight.available("qdrant"):
        raise RuntimeError("Isolated demo setup requires both explicitly selected services to pass preflight")
    if not preflight.available("embeddings"):
        raise RuntimeError("Isolated real indexing requires a locally loadable embedding model")
    setup = IsolatedDemoSetup(config, preflight, output_dir)
    with isolated_postgres(config.database_url) as stack:
        setup.bind_postgres(stack)
        try:
            setup.index()
            setup.update_rows()
            setup.write_audit()
            yield setup
        finally:
            setup.cleanup()
            setup.write_audit()
