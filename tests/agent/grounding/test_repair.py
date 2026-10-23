"""One regeneration can correct grounding; repeated invalid output remains a failure."""

from types import SimpleNamespace

import pytest

from agent.evidence import EvidenceValidationError
from agent.planner import StructuredAgentFinalizer
from claims.llm import LlmError, OllamaStructuredLlm
from tests.agent.grounding._support import finalizer_state
from tests.agent.review._support import follow_up_evidence as follow_up_evidence
from tests.agent.review._support import stored_review as stored_review
from tests.infrastructure.support.agent_fakes import FakeStructuredLlm, make_evidence, make_recommendation


def test_valid_first_response_does_not_add_generation():
    recommendation = make_recommendation()
    llm = FakeStructuredLlm([recommendation])
    result = StructuredAgentFinalizer(llm).finalize(finalizer_state())
    assert result == recommendation and len(llm.calls) == 1


@pytest.mark.parametrize("failure", ["unknown_id", "wrong_version"])
def test_one_correction_uses_only_safe_feedback_and_original_evidence(failure):
    first, corrected = make_recommendation(), make_recommendation()
    if failure == "unknown_id":
        first.evidence_ids.append("provider-secret-invented-id")
    else:
        first.prompt_version = "provider-secret-old-version"
    llm = FakeStructuredLlm([first, corrected])
    result = StructuredAgentFinalizer(llm).finalize(finalizer_state())
    assert result == corrected and len(llm.calls) == 2
    assert llm.calls[1][0].startswith(llm.calls[0][0])
    assert "APPLICATION VALIDATION FEEDBACK" in llm.calls[1][0]
    assert "provider-secret" not in llm.calls[1][0]


@pytest.mark.parametrize("failure", ["unknown_id", "wrong_version", "uncited"])
def test_wrapped_ollama_schema_error_gets_one_correction(monkeypatch, failure):
    first, corrected = make_recommendation(), make_recommendation()
    if failure == "unknown_id":
        first.evidence_ids.append("provider-secret-id")
    elif failure == "wrong_version":
        first.prompt_version = "provider-secret-version"
    else:
        first.rationale[0].evidence_ids = []
    responses, calls = [first.model_dump_json(), corrected.model_dump_json()], []

    def chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=responses.pop(0)))

    monkeypatch.setattr("claims.llm.Client", lambda **kwargs: SimpleNamespace(chat=chat))
    result = StructuredAgentFinalizer(OllamaStructuredLlm("offline")).finalize(finalizer_state())
    feedback_prompt = calls[1]["messages"][-1]["content"]
    assert result == corrected and len(calls) == 2
    assert "APPLICATION VALIDATION FEEDBACK" in feedback_prompt
    assert "provider-secret" not in feedback_prompt


def test_missing_follow_up_citation_is_corrected_by_model_not_rewritten(follow_up_evidence):
    claim = make_evidence()
    first = make_recommendation([claim.evidence_id])
    corrected = make_recommendation([follow_up_evidence.evidence_id])
    llm = FakeStructuredLlm([first, corrected])
    result = StructuredAgentFinalizer(llm).finalize(finalizer_state(claim, follow_up_evidence))
    assert result == corrected and len(llm.calls) == 2
    assert first.rationale[0].evidence_ids == [claim.evidence_id]
    assert "FOLLOW_UP_ASSESSMENT_NOT_CITED" in llm.calls[1][0]


def test_repeated_invalid_output_is_rejected_without_a_third_call():
    first = make_recommendation(["provider-secret-id"])
    llm = FakeStructuredLlm([first, first, make_recommendation()])
    with pytest.raises(EvidenceValidationError) as caught:
        StructuredAgentFinalizer(llm).finalize(finalizer_state())
    assert caught.value.code == "UNKNOWN_EVIDENCE_ID"
    assert len(llm.calls) == 2 and len(llm.outputs) == 1


def test_initial_transport_error_is_not_retried():
    llm = FakeStructuredLlm([LlmError("provider-secret-connection")])
    with pytest.raises(LlmError):
        StructuredAgentFinalizer(llm).finalize(finalizer_state())
    assert len(llm.calls) == 1


def test_failed_correction_keeps_original_grounding_cause():
    llm = FakeStructuredLlm([make_recommendation(["provider-secret-id"]), LlmError("provider-secret-connection")])
    with pytest.raises(EvidenceValidationError) as caught:
        StructuredAgentFinalizer(llm).finalize(finalizer_state())
    assert caught.value.code == "UNKNOWN_EVIDENCE_ID" and len(llm.calls) == 2
