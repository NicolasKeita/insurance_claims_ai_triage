"""The restricted action schema participates in bounded grounding correction."""

from types import SimpleNamespace

import pytest

from agent.models import RecommendedNextAction
from agent.planner import StructuredAgentFinalizer
from claims.llm import OllamaStructuredLlm
from tests.agent.grounding._support import finalizer_state
from tests.agent.review._support import follow_up_evidence as follow_up_evidence
from tests.agent.review._support import stored_review as stored_review
from tests.infrastructure.support.agent_fakes import FakeStructuredLlm, make_recommendation


@pytest.mark.parametrize("transport", ["custom_adapter", "ollama"])
def test_incompatible_follow_up_can_be_corrected_without_rewriting(monkeypatch, transport, follow_up_evidence):
    first = make_recommendation([follow_up_evidence.evidence_id])
    first.recommended_next_action = RecommendedNextAction.CONTINUE_STANDARD_REVIEW
    corrected = make_recommendation([follow_up_evidence.evidence_id])
    llm, calls = FakeStructuredLlm([first, corrected]), []
    if transport == "ollama":
        responses = [first.model_dump_json(), corrected.model_dump_json()]

        def chat(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(message=SimpleNamespace(content=responses.pop(0)))

        monkeypatch.setattr("claims.llm.Client", lambda **kwargs: SimpleNamespace(chat=chat))
        llm = OllamaStructuredLlm("offline")
    result = StructuredAgentFinalizer(llm).finalize(finalizer_state(follow_up_evidence))
    prompts = [call["messages"][-1]["content"] for call in calls] if calls else [call[0] for call in llm.calls]
    assert result == corrected and len(prompts) == 2
    assert "INCOMPATIBLE_FOLLOW_UP" in prompts[1]
    assert first.recommended_next_action == RecommendedNextAction.CONTINUE_STANDARD_REVIEW
