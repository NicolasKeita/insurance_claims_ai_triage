"""The post-loading eligibility check reads results without rerunning probes."""

from dataclasses import replace

from evaluation.preflight import run_preflight
from evaluation.preflight.prerequisites import PrerequisiteChecks
from evaluation.preflight.session import PreflightSession
from tests.infrastructure.prerequisites._evaluation import gate as gate
from tests.infrastructure.prerequisites._evaluation import ready_gate as ready_gate


def test_service_and_model_probes_run_once_despite_two_eligibility_checks(ready_gate, monkeypatch):
    config, scenario = ready_gate
    probes = []
    names = {
        "probe_postgres": ("postgresql",),
        "probe_qdrant": ("qdrant",),
        "probe_ollama": ("ollama",),
        "probe_collections": ("knowledge_collections", "similar_claims_collection"),
        "probe_embeddings": ("embeddings",),
        "probe_mlflow": ("mlflow",),
        "probe_models": ("triage", "anomaly"),
        "probe_detectors": (),
    }

    def fake_probe(method, components):
        def operation(session):
            probes.append(method)
            for component in components:
                session.probe(component, lambda: {"schema_available": True})

        return operation

    for method, components in names.items():
        monkeypatch.setattr(PreflightSession, method, fake_probe(method, components))
    monkeypatch.setattr(PrerequisiteChecks, "cached_embeddings", staticmethod(lambda models: {}))
    report = run_preflight(
        replace(config, include_rag=True, include_similar=True, include_agent=True), scenarios=[scenario]
    )
    assert not report.blocked_scenarios
    assert sorted(probes) == sorted(names)
    assert sum(row["component"] == "prerequisites:controlled" for row in report.rows) == 1


def test_pipeline_constructor_reuses_supplied_preflight_without_new_probes(monkeypatch):
    from evaluation.preflight import PreflightReport, RealEvaluationConfig
    from evaluation.real.pipeline import RealClaimsEvaluationPipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("The pipeline repeated the supplied preflight")

    monkeypatch.setattr("evaluation.real.pipeline.run_preflight", forbidden)
    report = PreflightReport()
    pipeline = RealClaimsEvaluationPipeline(RealEvaluationConfig(), report)
    assert pipeline.preflight is report and pipeline.resources is report.resources
