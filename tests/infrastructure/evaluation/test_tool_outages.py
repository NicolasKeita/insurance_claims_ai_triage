"""Request-scoped real-agent fault wiring without a benchmark or external service."""

from types import SimpleNamespace

import pytest

from evaluation.models import PipelineObservation
from evaluation.preflight import PreflightReport, RealEvaluationConfig
from evaluation.real.context import ComponentRecorder
from evaluation.real.pipeline import RealClaimsEvaluationPipeline
from evaluation.real.stages.agent import AgentSteps
from evaluation.tool_outages import history_outage_session, tool_outage_reason
from tests.agent.persistence._support import make_agent_run


class AgentRecorder(AgentSteps, ComponentRecorder):
    def similar_service(self):
        raise AssertionError("Service loading must remain lazy")


def recorder(*, isolated=True, active=True):
    instance = AgentRecorder()
    instance.config = RealEvaluationConfig(isolated_demo_stack=isolated)
    instance.preflight = PreflightReport(
        rows=[{"component": name, "status": "PASS"} for name in ("postgresql", "ollama")]
    )
    instance.resources = {"llm_model": "offline-model", "session_factory": object()}
    if active:
        instance.resources["isolated_schema"] = "evaluation_e2e_owned"
    return instance


def run_with_outage(tools=("GET_HISTORY",)):
    return SimpleNamespace(
        case=SimpleNamespace(claim=SimpleNamespace(claim_id="CONTROLLED-1")),
        scenario=SimpleNamespace(
            scenario_id="controlled",
            inputs={"unavailable_tools": list(tools)},
            vision_required=False,
        ),
        observed=PipelineObservation(),
    )


def test_history_outage_is_supported_with_an_active_isolated_stack():
    instance, run = recorder(), run_with_outage()
    assert tool_outage_reason(instance.config, run.scenario) is None
    assert instance._skip_agent(run.observed, run.scenario) is False
    assert not run.observed.components


@pytest.mark.parametrize("isolated, active", [(False, True), (True, False)])
def test_outage_requires_isolation_before_any_pipeline_stage(isolated, active):
    instance, run = recorder(isolated=isolated, active=active), run_with_outage()
    assert RealClaimsEvaluationPipeline._skip_prerequisites(instance, run) is True
    assert set(run.observed.components) == {"prerequisites"}
    assert run.observed.components["prerequisites"].status == "SKIP"
    assert "isolated" in run.observed.components["prerequisites"].reason
    assert "agent_tool_outages" not in run.observed.provenance


def test_unknown_tool_is_rejected_even_if_a_caller_bypasses_preflight():
    instance, run = recorder(), run_with_outage(["GET_CLAIM"])
    assert RealClaimsEvaluationPipeline._skip_prerequisites(instance, run) is True
    assert "Unsupported controlled agent tool outages" in run.observed.components["prerequisites"].reason


def test_history_connection_failure_is_wired_only_to_the_selected_agent(monkeypatch):
    instance, run = recorder(), run_with_outage()
    result, captured, closed = make_agent_run("CONTROLLED-1"), [], []

    def create(factory, **kwargs):
        assert factory is instance.resources["session_factory"]
        captured.append(kwargs)
        return SimpleNamespace(tools=SimpleNamespace(close=lambda: closed.append(True)))

    monkeypatch.setattr("agent.factory.create_claims_agent_service", create)
    monkeypatch.setattr(instance, "review", lambda *args: result)
    instance._run_agent(run, run.observed)

    assert captured[0]["history_session_factory"] is history_outage_session
    assert captured[0]["require_history"] is True
    assert run.observed.provenance["agent_tool_outages"] == {
        "controlled": True,
        "scope": "agent_only",
        "requested_tools": ["GET_HISTORY"],
        "isolated_schema": "evaluation_e2e_owned",
        "methods": {"GET_HISTORY": "injected_history_connection_failure"},
    }
    assert run.observed.trace[-1].inputs["controlled_unavailable_tools"] == ["GET_HISTORY"]
    assert run.observed.components["agent"].status == "PASS"
    assert closed == [True]

    normal_run = run_with_outage([])
    instance._run_agent(normal_run, normal_run.observed)
    assert "history_session_factory" not in captured[1]
    assert "require_history" not in captured[1]
    assert "agent_tool_outages" not in normal_run.observed.provenance
    assert closed == [True, True]


def test_runtime_fault_wiring_refuses_a_missing_isolated_schema(monkeypatch):
    instance, run = recorder(active=False), run_with_outage()

    def forbid_service(*args, **kwargs):
        raise AssertionError("Agent construction must not happen before isolation is checked")

    monkeypatch.setattr("agent.factory.create_claims_agent_service", forbid_service)
    with pytest.raises(ValueError, match="active isolated schema"):
        instance._run_agent(run, run.observed)
    assert "agent_tool_outages" not in run.observed.provenance
