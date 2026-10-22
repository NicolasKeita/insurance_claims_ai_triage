"""Unit eligibility checks; no benchmark scenario, service or model is executed."""

from dataclasses import replace

import pytest

from evaluation.preflight import run_preflight
from evaluation.preflight.prerequisites import PrerequisiteChecks
from evaluation.preflight.session import PreflightSession
from tests.infrastructure.prerequisites._evaluation import gate as gate
from tests.infrastructure.prerequisites._evaluation import ready_gate as ready_gate


def test_missing_database_skips_before_loading_any_models(gate):
    config, scenario = gate
    report = run_preflight(config, scenarios=[scenario])
    assert "DATABASE_URL is not configured" in "; ".join(report.blocked_scenarios[scenario.scenario_id])
    assert not any(row["component"] in {"triage", "anomaly", "embeddings"} for row in report.rows)


@pytest.mark.parametrize("missing", ["source", "checkpoint"])
def test_missing_required_files_skip_before_loading(ready_gate, tmp_path, missing):
    config, scenario = ready_gate
    if missing == "source":
        scenario = scenario.model_copy(update={"claim_source": str(tmp_path / "absent.json")})
    else:
        config = replace(config, triage_checkpoint=tmp_path / "absent.joblib")
    report = run_preflight(config, scenarios=[scenario])
    assert "Required file missing or empty" in "; ".join(report.blocked_scenarios[scenario.scenario_id])


def test_preflight_only_never_deserializes_existing_files(ready_gate):
    config, scenario = ready_gate
    report = run_preflight(config, scenarios=[scenario], load_models=False)
    assert not report.blocked_scenarios
    assert report.available("triage_files") and not report.available("triage")


def test_disabled_search_does_not_require_its_configuration_or_services(ready_gate, monkeypatch):
    config, scenario = ready_gate
    monkeypatch.setenv("RAG_INDEX_BATCH_SIZE", "invalid-unused-setting")
    monkeypatch.setattr(PreflightSession, "probe_qdrant", lambda self: pytest.fail("Unexpected Qdrant probe"))
    report = run_preflight(config, scenarios=[scenario], load_models=False)
    assert not report.blocked_scenarios


def test_unreachable_qdrant_blocks_enabled_search_before_loading(ready_gate, monkeypatch):
    config, scenario = ready_gate

    def qdrant(session):
        def unreachable():
            raise TimeoutError("Qdrant did not respond")

        session.probe("qdrant", unreachable)

    monkeypatch.setattr(PreflightSession, "probe_qdrant", qdrant)
    monkeypatch.setattr(PrerequisiteChecks, "cached_embeddings", staticmethod(lambda models: {}))
    report = run_preflight(replace(config, include_similar=True), scenarios=[scenario])
    assert "Qdrant did not respond" in "; ".join(report.blocked_scenarios[scenario.scenario_id])


def test_ollama_reachable_but_model_missing_blocks_agent(ready_gate, monkeypatch):
    import httpx

    config, scenario = ready_gate

    def tags(url, **kwargs):
        return httpx.Response(200, json={"models": [{"name": "different:latest"}]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", tags)
    report = run_preflight(replace(config, include_agent=True, llm_model="absent"), scenarios=[scenario])
    assert "is not installed" in "; ".join(report.blocked_scenarios[scenario.scenario_id])


def test_mixed_selection_loads_only_after_all_light_checks(ready_gate, monkeypatch, tmp_path):
    config, valid = ready_gate
    invalid = valid.model_copy(update={"scenario_id": "missing", "files": [str(tmp_path / "absent")]})
    seen = []

    def models(session):
        assert "missing" in session.report.blocked_scenarios
        seen.append("models")
        for name in ("triage", "anomaly"):
            if not session.report.available(name):
                session.probe(name, lambda: {})

    for name in ("probe_embeddings", "probe_models", "probe_detectors", "probe_mlflow"):
        monkeypatch.setattr(PreflightSession, name, models)
    report = run_preflight(config, scenarios=[invalid, valid])
    assert set(report.blocked_scenarios) == {"missing"}
    assert seen == ["models"] * 4


def test_empty_database_is_blocked_without_isolated_setup(gate, monkeypatch):
    config, scenario = gate
    monkeypatch.setattr(PreflightSession, "probe_postgres", lambda self: self.probe("postgresql", lambda: {}))
    report = run_preflight(config, scenarios=[scenario])
    assert "claims schema missing" in "; ".join(report.blocked_scenarios[scenario.scenario_id])


def test_isolated_setup_does_not_require_preexisting_claims_or_collections(ready_gate, monkeypatch):
    config, scenario = ready_gate
    config = replace(
        config,
        isolated_demo_stack=True,
        database_url="postgresql+psycopg://claims_test:claims_test@127.0.0.1:5434/insurance_claims_test",
        qdrant_url="http://127.0.0.1:6335",
        include_similar=True,
    )
    monkeypatch.setattr(PreflightSession, "probe_qdrant", lambda self: self.probe("qdrant", lambda: {}))
    monkeypatch.setattr(PrerequisiteChecks, "cached_embeddings", staticmethod(lambda models: {}))
    report = run_preflight(config, scenarios=[scenario], load_models=False)
    assert not report.blocked_scenarios
    assert not report.available("similar_claims_collection")
