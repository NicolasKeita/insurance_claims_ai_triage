"""Scenario eligibility checks performed before any model is loaded."""

from evaluation.preflight.files import embedding_files, require_file, scenario_files, trained_model_files
from evaluation.preflight.report import record_eligibility, requirements


class PrerequisiteChecks:
    """Reuse the service probes and their clients, but defer all model loading."""

    def __init__(self, session):
        self.session = session
        self.config = session.config

    def check(self, scenarios):
        session = self.session
        required = {item.scenario_id: requirements(self.config, item) for item in scenarios}
        names = set().union(*required.values())
        retrieval = names & {"qdrant", "embeddings", "knowledge_collections", "similar_claims_collection"}
        session.probe("retrieval_configuration", session.load_configs if retrieval else lambda: {})
        if "isolated_configuration" in names:
            session.probe("isolated_configuration", self.isolated_configuration)
        if "postgresql" in names:
            session.probe_postgres()
        if "qdrant" in names and session.report.available("retrieval_configuration"):
            session.probe_qdrant()
            session.probe_collections()
        if "ollama" in names:
            session.probe_ollama()
        for scenario in scenarios:
            session.probe(f"scenario_files:{scenario.scenario_id}", lambda item=scenario: scenario_files(item))
            if "postgresql" in required[scenario.scenario_id] and not self.config.isolated_demo_stack:
                session.probe(f"stored_claim:{scenario.scenario_id}", lambda item=scenario: self.stored_claim(item))
        self.check_model_files(names)
        for scenario in scenarios:
            record_eligibility(session, scenario, required[scenario.scenario_id])

    def isolated_configuration(self):
        from evaluation.preflight.config import ROOT
        from persistence.database import require_test_database

        if not self.config.database_url or not self.config.qdrant_url:
            raise RuntimeError("--isolated-demo-stack requires explicit --database-url and --qdrant-url")
        require_test_database(self.config.database_url)
        require_file(ROOT / "alembic.ini")
        require_file(ROOT / "migrations/env.py")
        if self.config.include_rag:
            from retrieval.knowledge.sources import KnowledgeCatalog

            settings = self.session.knowledge_config
            KnowledgeCatalog.load(settings.knowledge_dir, settings.max_chunk_chars)

    def stored_claim(self, scenario):
        from persistence.repositories import ClaimRepository

        report = self.session.report
        if not report.available("postgresql"):
            raise RuntimeError("PostgreSQL is unavailable")
        row = next(row for row in report.rows if row["component"] == "postgresql")
        if not row.get("schema_available"):
            raise RuntimeError("PostgreSQL claims schema missing; prepare it or use --isolated-demo-stack")
        claim_id = scenario.inputs.get("claim_id")
        if scenario.inputs.get("use_reference_case"):
            import json

            from evaluation.preflight.config import ROOT

            source = ROOT / scenario.claim_source
            source = source / "claim.json" if source.is_dir() else source
            claim_id = json.loads(require_file(source).read_text(encoding="utf-8"))["claim_id"]
        with report.resources["session_factory"]() as session:
            claim = ClaimRepository(session).get_by_claim_id(claim_id)
        if claim is None:
            raise RuntimeError(f"Controlled claim {claim_id!r} is not seeded; use --isolated-demo-stack")

    def check_model_files(self, names):
        session = self.session
        for kind in ("triage", "anomaly"):
            if kind in names:
                session.probe(
                    f"{kind}_files",
                    lambda name=kind: trained_model_files(name, getattr(self.config, name + "_checkpoint")),
                )
        for kind, setting in (("damage_vision", "damage_checkpoint"), ("part_vision", "part_checkpoint")):
            if kind in names:
                session.probe(f"{kind}_files", lambda key=setting: self.vision_file(key))
        if "embeddings" in names and session.report.available("retrieval_configuration"):
            models = {session.knowledge_config.embedding_model}
            if self.config.include_similar:
                models.add(session.retrieval_config.embedding_model)
            session.probe(
                "embedding_files",
                lambda: self.cached_embeddings(models),
                enabled=self.config.include_rag or self.config.include_similar,
            )

    @staticmethod
    def cached_embeddings(models):
        for model in sorted(models):
            embedding_files(model)
        return {"models": sorted(models), "meaning": "Local files present; models not loaded"}

    def vision_file(self, setting):
        if not self.config.include_vision:
            raise RuntimeError("Vision required by the scenario but disabled by configuration")
        return {"checkpoint": str(require_file(getattr(self.config, setting)))}
