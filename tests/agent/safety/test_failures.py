"""Test failures."""

from types import SimpleNamespace

import pytest
from sqlalchemy.exc import SQLAlchemyError

from agent.models import RunStatus, ToolObservation
from agent.service import AgentRunError
from tests.agent.safety._constants import CLAIM_ID
from tests.agent.safety._support import make_service
from tests.infrastructure.support.agent_fakes import (
    FakeAgentTools,
    FakeRunStore,
    make_recommendation,
)


def test_application_limits_survive_twenty_model_uncertainties():
    tools = FakeAgentTools(
        observations={
            "GET_ANOMALY": ToolObservation(
                tool_name="GET_ANOMALY", status="NOT_AVAILABLE", message="No stored anomaly assessment"
            )
        }
    )
    recommendation = make_recommendation()
    recommendation.uncertainties = [f"Model uncertainty {number}" for number in range(20)]
    service = make_service(["GET_CLAIM", "GET_ANOMALY"], tools=tools, recommendation=recommendation, max_iterations=2)
    result = service.review_claim(CLAIM_ID)
    uncertainties = result.recommendation.uncertainties
    assert len(uncertainties) == 20
    assert uncertainties[:2] == ["GET_ANOMALY: No stored anomaly assessment", "Maximum tool steps reached."]
    assert "Model uncertainty 0" in uncertainties


def test_unexpected_graph_exception_persists_sanitized_failed_run():
    store = FakeRunStore()
    service = make_service([], store=store)

    def broken_runtime(*args, **kwargs):
        raise RuntimeError("secret provider credential and raw document")

    service.graph = SimpleNamespace(invoke=broken_runtime)
    with pytest.raises(AgentRunError) as caught:
        service.review_claim(CLAIM_ID)
    assert caught.value.failure_kind == "ORCHESTRATION"
    assert len(store.results) == 1
    result = store.results[0]
    assert result.run_id == caught.value.run_id
    assert result.status == RunStatus.FAILED
    assert result.recommendation is None
    assert result.iteration_count == result.tool_call_count == 0
    assert result.error == "Agent orchestration failed."
    assert "secret" not in result.model_dump_json()


def test_orchestration_failure_does_not_hide_failed_audit_persistence():
    class BrokenStore:
        def save(self, run):
            assert run.status == RunStatus.FAILED
            raise SQLAlchemyError("Audit insert failed")

    service = make_service([], store=BrokenStore())

    def broken_runtime(*args, **kwargs):
        raise RuntimeError("Graph runtime failure")

    service.graph = SimpleNamespace(invoke=broken_runtime)
    with pytest.raises(SQLAlchemyError):
        service.review_claim(CLAIM_ID)
