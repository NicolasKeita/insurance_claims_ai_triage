"""Isolated indexing reuses preflight weights, preserving distinct model choices."""

import pytest

from evaluation.isolated_stack.setup import IsolatedDemoSetup
from evaluation.preflight import RealEvaluationConfig
from evaluation.preflight.prerequisites import PrerequisiteChecks
from evaluation.preflight.session import PreflightSession
from evaluation.real.similar_claims import SimilarClaimsSteps
from tests.infrastructure.prerequisites.loading._support import model_loads as model_loads


@pytest.mark.parametrize("claims_model", ["shared", "claims-only"])
def test_isolated_indexing_reuses_the_preflight_provider_when_models_match(
    monkeypatch, tmp_path, model_loads, claims_model
):
    from retrieval.knowledge.indexing import KnowledgeIndexer
    from retrieval.qdrant_store import QdrantClaimStore

    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "shared")
    monkeypatch.setenv("CLAIMS_EMBEDDING_MODEL", claims_model)
    session = PreflightSession(RealEvaluationConfig(include_vision=False))
    session.load_configs()
    session.probe_embeddings()
    session.report.resources["qdrant_client"] = object()
    monkeypatch.setattr(KnowledgeIndexer, "index_all", lambda *args: {})
    monkeypatch.setattr(QdrantClaimStore, "ensure_collection", lambda store: store.embeddings.dimension)
    setup = IsolatedDemoSetup(session.config, session.report, tmp_path)
    setup.index()
    resources = session.report.resources
    rag, claims = resources["embeddings"], resources["similar_embeddings"]
    steps = SimilarClaimsSteps()
    steps.resources = resources
    assert steps.similar_embeddings(resources["retrieval_config"]) is claims
    assert claims.model_name == claims_model and claims.asymmetric is False
    assert rag.asymmetric is True
    assert (rag._model is claims._model) == (claims_model == "shared")
    expected = ["shared"] if claims_model == "shared" else ["shared", "claims-only"]
    assert [name for name, _ in model_loads] == expected
    assert all(options == {"local_files_only": True} for _, options in model_loads)


def test_similar_only_preflight_does_not_load_or_require_unused_rag_model(monkeypatch, model_loads):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "unused-rag-model")
    monkeypatch.setenv("CLAIMS_EMBEDDING_MODEL", "claims-only")
    session = PreflightSession(RealEvaluationConfig(include_rag=False))
    session.probe("retrieval_configuration", session.load_configs)
    checked = []
    monkeypatch.setattr("evaluation.preflight.prerequisites.embedding_files", lambda model: checked.append(model))
    PrerequisiteChecks(session).check_model_files({"embeddings"})
    session.probe_embeddings()
    assert checked == ["claims-only"]
    assert model_loads == [("claims-only", {"local_files_only": True})]
    assert session.report.resources["similar_embeddings"]._model is session.report.resources["embeddings"]._model
