"""Read-only dependency probes; heavy adapters are loaded once and retained."""

from collections.abc import Callable
from dataclasses import replace
from time import perf_counter

from evaluation.preflight.components import ComponentProbes
from evaluation.preflight.config import RealEvaluationConfig
from evaluation.preflight.models import ModelProbes
from evaluation.preflight.report import PreflightReport


class PreflightSession(ComponentProbes, ModelProbes):
    """Run every probe against one report and keep successfully loaded adapters."""

    def __init__(self, config: RealEvaluationConfig, progress: Callable[[dict], None] | None = None):
        self.config = config
        self.progress = progress
        self.report = PreflightReport()
        self.knowledge_config = None
        self.retrieval_config = None

    def probe(self, name, operation, *, enabled=True):
        """Missing optional dependencies are explicit SKIP/FAIL according to config."""
        start = perf_counter()
        row = {"component": name, "status": "SKIP", "version": None, "reason": "Disabled by configuration"}
        if enabled:
            try:
                metadata = operation() or {}
                row.update(status="PASS", reason=None, **metadata)
            except Exception as error:
                row.update(
                    status="SKIP" if name == "mlflow" else self.config.dependency_unavailable,
                    reason=f"{type(error).__name__}: {error}",
                )
        row["duration_seconds"] = round(perf_counter() - start, 6)
        self.report.rows.append(row)
        if self.progress is not None:
            self.progress(row)
        return row

    def run(self) -> PreflightReport:
        """Probe every configured dependency in a fixed, reviewable order."""
        self.probe_postgres()
        self.probe_qdrant()
        self.load_configs()
        self.probe_collections()
        self.probe_embeddings()
        self.probe_ollama()
        self.probe_mlflow()
        self.probe_models()
        self.probe_detectors()
        return self.report

    def load_configs(self):
        from knowledge.config import KnowledgeConfig
        from retrieval.config import RetrievalConfig

        self.knowledge_config = KnowledgeConfig.from_env()
        self.retrieval_config = RetrievalConfig.from_env()
        if self.config.qdrant_url:
            self.knowledge_config = replace(self.knowledge_config, qdrant_url=self.config.qdrant_url)
            self.retrieval_config = replace(self.retrieval_config, qdrant_url=self.config.qdrant_url)
        self.report.resources.update(
            knowledge_config=self.knowledge_config, retrieval_config=self.retrieval_config
        )

    def collections(self, names):
        if not self.report.available("qdrant"):
            raise RuntimeError("Qdrant is unavailable")
        missing = set(names) - set(self.report.resources["collections"])
        if missing:
            raise RuntimeError("Missing collections: " + ", ".join(sorted(missing)))
        client = self.report.resources["qdrant_client"]
        return {
            "collections": list(names),
            "point_counts": {name: client.get_collection(name).points_count for name in names},
        }

    def probe_collections(self):
        self.probe(
            "knowledge_collections",
            lambda: self.collections(
                [self.knowledge_config.policy_collection, self.knowledge_config.procedure_collection]
            ),
            enabled=self.config.include_rag,
        )
        self.probe(
            "similar_claims_collection",
            lambda: self.collections([self.retrieval_config.collection]),
            enabled=self.config.include_similar,
        )

    def probe_embeddings(self):
        def embeddings():
            """local_files_only is enforced even when another import initialized HF settings."""
            from retrieval.embeddings import SentenceTransformerEmbeddingProvider

            provider = SentenceTransformerEmbeddingProvider(
                self.knowledge_config.embedding_model, asymmetric=True, local_files_only=True
            )
            self.report.resources["embeddings"] = provider
            return {
                "version": provider.model_name,
                "dimension": provider.dimension,
                "local_files_only": True,
            }

        self.probe("embeddings", embeddings, enabled=self.config.include_rag or self.config.include_similar)
