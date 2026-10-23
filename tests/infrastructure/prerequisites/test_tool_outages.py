"""Controlled tool outages must be feasible before loading evaluation models."""

from dataclasses import replace

import pytest

from evaluation.preflight import run_preflight
from evaluation.preflight.prerequisites import PrerequisiteChecks
from evaluation.preflight.session import PreflightSession
from tests.infrastructure.prerequisites._evaluation import gate as gate
from tests.infrastructure.prerequisites._evaluation import ready_gate as ready_gate


@pytest.fixture
def outage_gate(ready_gate, monkeypatch):
    config, scenario = ready_gate
    config = replace(
        config,
        include_agent=True,
        isolated_demo_stack=True,
        database_url="postgresql+psycopg://claims_test:claims_test@127.0.0.1:5434/insurance_claims_test",
        qdrant_url="http://127.0.0.1:6335",
    )
    scenario = scenario.model_copy(update={"inputs": {**scenario.inputs, "unavailable_tools": ["GET_HISTORY"]}})
    monkeypatch.setattr(PreflightSession, "probe_ollama", lambda self: self.probe("ollama", lambda: {}))
    monkeypatch.setattr(PreflightSession, "probe_qdrant", lambda self: self.probe("qdrant", lambda: {}))
    monkeypatch.setattr(PrerequisiteChecks, "cached_embeddings", staticmethod(lambda models: {}))
    return config, scenario


@pytest.mark.parametrize(
    "tools",
    [["GET_HISTORY"], ["GET_INVESTIGATION"], ["GET_VISUAL_ASSESSMENT", "GET_HISTORY", "GET_HISTORY"]],
)
def test_supported_isolated_outages_are_preflighted_before_model_files(outage_gate, tools):
    config, scenario = outage_gate
    scenario = scenario.model_copy(update={"inputs": {**scenario.inputs, "unavailable_tools": tools}})

    report = run_preflight(config, scenarios=[scenario], load_models=False)

    assert not report.blocked_scenarios
    name = f"scenario_agent_tools:{scenario.scenario_id}"
    row = next(item for item in report.rows if item["component"] == name)
    assert row["status"] == "PASS"
    assert row["unavailable_tools"] == sorted(set(tools))
    assert row["isolated_demo_stack"] is True
    assert "no tool executed" in row["meaning"]
    names = [item["component"] for item in report.rows]
    assert names.index(f"scenario_files:{scenario.scenario_id}") < names.index(name) < names.index("triage_files")
    assert not report.available("triage")


@pytest.mark.parametrize("policy", ["SKIP", "FAIL"])
@pytest.mark.parametrize(
    "change,reason",
    [
        ("nonisolated", "require --isolated-demo-stack"),
        ("disabled_agent", "require --include-agent"),
        ("unsupported_tool", "Unsupported controlled agent tool outages: GET_CLAIM"),
    ],
)
def test_unverifiable_outages_block_before_any_models_load(outage_gate, policy, change, reason):
    config, scenario = outage_gate
    config = replace(config, dependency_unavailable=policy)
    if change == "nonisolated":
        config = replace(config, isolated_demo_stack=False)
    elif change == "disabled_agent":
        config = replace(config, include_agent=False)
    else:
        scenario = scenario.model_copy(update={"inputs": {**scenario.inputs, "unavailable_tools": ["GET_CLAIM"]}})

    report = run_preflight(config, scenarios=[scenario])

    name = f"scenario_agent_tools:{scenario.scenario_id}"
    row = next(item for item in report.rows if item["component"] == name)
    eligibility = next(item for item in report.rows if item["component"] == f"prerequisites:{scenario.scenario_id}")
    assert row["status"] == eligibility["status"] == policy
    assert reason in row["reason"]
    assert name in eligibility["missing_components"]
    assert reason in "; ".join(report.blocked_scenarios[scenario.scenario_id])
    assert not any(item["component"] in {"triage", "anomaly", "embeddings"} for item in report.rows)


def test_scenario_without_outages_does_not_require_an_agent_tool_probe(ready_gate):
    config, scenario = ready_gate

    report = run_preflight(config, scenarios=[scenario], load_models=False)

    assert not report.blocked_scenarios
    assert not any(item["component"].startswith("scenario_agent_tools:") for item in report.rows)


def test_mixed_selection_retains_outage_block_after_other_models_load(outage_gate, monkeypatch):
    config, invalid = outage_gate
    invalid = invalid.model_copy(update={"inputs": {**invalid.inputs, "unavailable_tools": ["GET_CLAIM"]}})
    valid = invalid.model_copy(update={"scenario_id": "ready", "inputs": {"claim_id": "CONTROLLED-1"}})
    loads = []

    def models(session):
        assert set(session.report.blocked_scenarios) == {invalid.scenario_id}
        assert not session.report.available(f"scenario_agent_tools:{invalid.scenario_id}")
        loads.append("models")
        for name in ("triage", "anomaly", "embeddings"):
            if not session.report.available(name):
                session.probe(name, lambda: {})

    for name in ("probe_embeddings", "probe_models", "probe_detectors", "probe_mlflow"):
        monkeypatch.setattr(PreflightSession, name, models)

    report = run_preflight(config, scenarios=[invalid, valid])

    assert set(report.blocked_scenarios) == {invalid.scenario_id}
    assert loads == ["models"] * 4
    assert report.available("prerequisites:ready")
    assert sum(item["component"] == f"scenario_agent_tools:{invalid.scenario_id}" for item in report.rows) == 1
