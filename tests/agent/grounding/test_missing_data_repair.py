"""Missing factual data constrains model decisions without replacing its answer."""

import json
from types import SimpleNamespace

import pytest

from agent.evidence import EvidenceValidationError
from agent.models import EvidenceSourceType, RecommendedNextAction
from agent.planner import StructuredAgentFinalizer
from claims.llm import OllamaStructuredLlm
from tests.agent.grounding._support import finalizer_state
from tests.infrastructure.support.agent_fakes import make_evidence, make_recommendation

EVIDENCE_MARKER = "UNTRUSTED TASK AND TOOL EVIDENCE (JSON):\n"
FEEDBACK_MARKER = "\n\nAPPLICATION VALIDATION FEEDBACK:\n"


def snapshot(value):
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def missing_history_state(status="NOT_AVAILABLE"):
    state = finalizer_state()
    state["observations"] = [
        {"tool_name": "GET_HISTORY", "status": status, "data": None, "evidence": [], "message": "History unavailable"}
    ]
    return state


def unchecked_llm(monkeypatch, transport, outputs):
    pending, calls = list(outputs), []
    if transport == "ollama":

        def chat(**kwargs):
            calls.append(kwargs["messages"][-1]["content"])
            return SimpleNamespace(message=SimpleNamespace(content=pending.pop(0).model_dump_json()))

        monkeypatch.setattr("claims.llm.Client", lambda **kwargs: SimpleNamespace(chat=chat))
        return OllamaStructuredLlm("offline"), pending, calls

    class UncheckedLlm:
        def generate(self, *, prompt, response_model):
            calls.append(prompt)
            # Deliberately bypass the provider's response_model: the application must revalidate it.
            return pending.pop(0)

    return UncheckedLlm(), pending, calls


def private_candidate(ids, action=RecommendedNextAction.CONTINUE_STANDARD_REVIEW):
    candidate = make_recommendation(ids)
    candidate.recommended_next_action = action
    candidate.case_summary = "provider-secret-draft-summary"
    candidate.rationale[0].statement = "provider-secret-draft-rationale"
    candidate.uncertainties = ["provider-secret-draft-uncertainty"]
    return candidate


def assert_safe_correction(prompts, code):
    assert len(prompts) == 2
    assert prompts[1].startswith(prompts[0])
    original, feedback = prompts[1].split(FEEDBACK_MARKER, 1)
    assert original == prompts[0]
    assert json.loads(feedback)["code"] == code
    assert "provider-secret" not in prompts[1]
    assert "RAW LLM RESPONSE" not in feedback
    assert "PYDANTIC ERROR" not in feedback
    assert len(feedback) < 1000


@pytest.mark.parametrize("transport", ["custom_adapter", "ollama"])
@pytest.mark.parametrize("status", ["NOT_AVAILABLE", "NOT_FOUND"])
@pytest.mark.parametrize("action", ["REQUEST_ADDITIONAL_INFORMATION", "NO_RECOMMENDATION"])
def test_missing_history_action_is_corrected_once_without_mutating_evidence(monkeypatch, transport, status, action):
    state = missing_history_state(status)
    ids = list(state["evidence_registry"])
    first = private_candidate(ids)
    corrected = make_recommendation(ids)
    corrected.recommended_next_action = RecommendedNextAction(action)
    original_state, original_first, original_corrected = snapshot(state), snapshot(first), snapshot(corrected)
    llm, pending, prompts = unchecked_llm(monkeypatch, transport, [first, corrected])

    result = StructuredAgentFinalizer(llm).finalize(state)

    assert result == corrected
    assert result.recommended_next_action == action
    assert not pending
    assert snapshot(state) == original_state
    assert snapshot(first) == original_first
    assert snapshot(corrected) == original_corrected
    assert_safe_correction(prompts, "INCOMPATIBLE_MISSING_DATA")
    context = json.loads(prompts[0].split(EVIDENCE_MARKER, 1)[1])
    assert set(context["allowed_next_actions"]) == {"REQUEST_ADDITIONAL_INFORMATION", "NO_RECOMMENDATION"}
    assert context["allowed_evidence_ids"] == ids
    assert context["observations"] == [{"tool": "GET_HISTORY", "status": status, "message": "History unavailable"}]


@pytest.mark.parametrize("transport", ["custom_adapter", "ollama"])
@pytest.mark.parametrize("action", ["CONTINUE_STANDARD_REVIEW", "REFER_TO_EXPERT"])
def test_two_invalid_missing_data_answers_fail_without_a_third_call_or_replacement(monkeypatch, transport, action):
    state = missing_history_state()
    ids = list(state["evidence_registry"])
    first = private_candidate(ids, RecommendedNextAction(action))
    second = first.model_copy(deep=True)
    second.case_summary = "provider-secret-second-draft-summary"
    unused = make_recommendation(ids)
    original_state = snapshot(state)
    original_candidates = [snapshot(item) for item in (first, second, unused)]
    llm, pending, prompts = unchecked_llm(monkeypatch, transport, [first, second, unused])

    with pytest.raises(EvidenceValidationError) as caught:
        StructuredAgentFinalizer(llm).finalize(state)

    assert caught.value.code == "INCOMPATIBLE_MISSING_DATA"
    assert "provider-secret" not in str(caught.value)
    assert pending == [unused]
    assert snapshot(state) == original_state
    assert [snapshot(item) for item in (first, second, unused)] == original_candidates
    assert_safe_correction(prompts, "INCOMPATIBLE_MISSING_DATA")


@pytest.mark.parametrize("transport", ["custom_adapter", "ollama"])
def test_positive_assessment_retains_follow_up_priority_when_history_is_missing(monkeypatch, transport):
    state = missing_history_state()
    assessment = make_evidence("investigation:stored").model_copy(
        update={
            "source_type": EvidenceSourceType.INVESTIGATION_ASSESSMENT,
            "structured_data": {"signals": [{"code": "VISUAL_ADDITIONAL_DAMAGE"}]},
        }
    )
    state["evidence_registry"][assessment.evidence_id] = assessment.model_dump(mode="json")
    ids = list(state["evidence_registry"])
    first = private_candidate(ids, RecommendedNextAction.NO_RECOMMENDATION)
    corrected = make_recommendation(ids)
    corrected.recommended_next_action = RecommendedNextAction.REFER_TO_EXPERT
    original_state, original_first = snapshot(state), snapshot(first)
    llm, pending, prompts = unchecked_llm(monkeypatch, transport, [first, corrected])

    result = StructuredAgentFinalizer(llm).finalize(state)

    assert result == corrected
    assert not pending
    assert snapshot(state) == original_state
    assert snapshot(first) == original_first
    assert_safe_correction(prompts, "INCOMPATIBLE_FOLLOW_UP")
    context = json.loads(prompts[0].split(EVIDENCE_MARKER, 1)[1])
    assert set(context["allowed_next_actions"]) == {
        "REQUEST_ADDITIONAL_INFORMATION",
        "REFER_TO_EXPERT",
        "REFER_FOR_INVESTIGATION_REVIEW",
    }
    assert context["required_follow_up_evidence_ids"] == [assessment.evidence_id]
