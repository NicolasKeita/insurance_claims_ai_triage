from types import SimpleNamespace
import json

import pytest
from sqlalchemy.exc import SQLAlchemyError

from agent.models import RunStatus, ToolObservation
from agent.config import MAX_FINALIZER_EVIDENCE_CHARS
from agent.planner import StructuredAgentFinalizer, StructuredAgentPlanner
from claims.llm import OllamaStructuredLlm
from agent.service import AgentRunError, ClaimsAgentService
from agent_fakes import (
    FakeAgentTools, FakeRunStore, FakeStructuredLlm, make_evidence, make_recommendation, make_step,
)

CLAIM_ID = "CLAIM-2026-00001"


def make_service(steps, *, tools=None, recommendation=None, store=None, max_iterations=8):
    return ClaimsAgentService(
        tools or FakeAgentTools(),
        StructuredAgentPlanner(FakeStructuredLlm([make_step(step) for step in steps])),
        StructuredAgentFinalizer(FakeStructuredLlm([recommendation or make_recommendation()])),
        run_store=store, max_iterations=max_iterations,
    )


def test_application_limits_survive_twenty_model_uncertainties():
    tools = FakeAgentTools(observations={"GET_ANOMALY": ToolObservation(
        tool_name="GET_ANOMALY", status="NOT_AVAILABLE", message="No stored anomaly assessment")})
    recommendation = make_recommendation()
    recommendation.uncertainties = [f"Model uncertainty {number}" for number in range(20)]
    service = make_service(["GET_CLAIM", "GET_ANOMALY"], tools=tools,
                           recommendation=recommendation, max_iterations=2)
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


def test_generated_versions_must_match_current_agent():
    recommendation = make_recommendation()
    recommendation.prompt_version = "unrecognized_prompt_version"
    service = make_service(["GET_CLAIM", "FINISH"], recommendation=recommendation)
    with pytest.raises(AgentRunError) as caught:
        service.review_claim(CLAIM_ID)
    assert caught.value.failure_kind == "GROUNDING"


def test_rationale_citations_are_aggregated_even_when_llm_omits_top_level_ids():
    recommendation = make_recommendation()
    recommendation.evidence_ids = []
    service = make_service(["GET_CLAIM", "FINISH"], recommendation=recommendation)
    result = service.review_claim(CLAIM_ID)
    assert result.recommendation.evidence_ids == [f"claim:{CLAIM_ID}"]


def test_omitted_finalizer_grounding_fields_are_invalid_generation():
    service = ClaimsAgentService(FakeAgentTools(),
        StructuredAgentPlanner(FakeStructuredLlm([make_step("GET_CLAIM"), make_step("FINISH")])),
        StructuredAgentFinalizer(FakeStructuredLlm([{
            "case_summary": "A claim needs review.", "recommended_next_action": "CONTINUE_STANDARD_REVIEW",
        }])))
    with pytest.raises(AgentRunError) as caught:
        service.review_claim(CLAIM_ID)
    assert caught.value.failure_kind == "GENERATION"


def test_large_finalizer_evidence_is_bounded_without_losing_ids():
    llm = FakeStructuredLlm([make_recommendation()])
    registry = {}
    for index in range(40):
        item = make_evidence(f"claim:context-{index}", summary='\\"' * 1900)
        item.structured_data = {"free_text": "x" * 12000}
        registry[item.evidence_id] = item.model_dump(mode="json")
    output = StructuredAgentFinalizer(llm).finalize({
        "claim_id": CLAIM_ID, "objective": "Review evidence", "stop_reason": "Maximum tool steps reached.",
        "evidence_registry": registry, "observations": [],
    })
    data = json.loads(llm.calls[0][0].split("UNTRUSTED TASK AND TOOL EVIDENCE (JSON):\n")[1])
    assert len(json.dumps(data["evidence"], ensure_ascii=False)) <= MAX_FINALIZER_EVIDENCE_CHARS
    assert {e["evidence_id"] for e in data["evidence"]} == set(registry)
    assert data["evidence_context_truncated"] is True
    assert "Evidence details were shortened to fit the model context." in output.uncertainties


def test_agent_context_window_is_forwarded_through_existing_llm(monkeypatch):
    calls = []

    class RecordingClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(message=SimpleNamespace(content=make_step("FINISH").model_dump_json()))

    monkeypatch.setattr("claims.llm.Client", RecordingClient)
    llm = OllamaStructuredLlm("fake", context_window=8192)
    llm.generate(prompt="Review task", response_model=type(make_step("FINISH")))
    assert calls[0]["options"]["num_ctx"] == 8192
    assert calls[0]["options"]["temperature"] == 0
    assert calls[0]["options"]["seed"] == 42
