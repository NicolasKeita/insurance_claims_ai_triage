"""The restricted action schema participates in bounded grounding correction."""

from types import SimpleNamespace

import pytest

from agent.models import EvidenceSourceType, RecommendedNextAction
from agent.planner import StructuredAgentFinalizer
from claims.llm import OllamaStructuredLlm
from tests.agent.grounding._support import finalizer_state
from tests.infrastructure.support.agent_fakes import FakeStructuredLlm, make_evidence, make_recommendation


@pytest.mark.parametrize("transport", ["custom_adapter", "ollama"])
@pytest.mark.parametrize("code", ["ADDITIONAL_QUOTE_DAMAGE", "RECENT_CLAIM_FREQUENCY"])
def test_incompatible_follow_up_can_be_corrected_without_rewriting(monkeypatch, transport, code):
    investigation = make_evidence("investigation:stored").model_copy(
        update={
            "source_type": EvidenceSourceType.INVESTIGATION_ASSESSMENT,
            "structured_data": {"signals": [{"code": code}]},
        }
    )
    first = make_recommendation([investigation.evidence_id])
    first.recommended_next_action = RecommendedNextAction.CONTINUE_STANDARD_REVIEW
    corrected = make_recommendation([investigation.evidence_id])
    llm, calls = FakeStructuredLlm([first, corrected]), []
    if transport == "ollama":
        responses = [first.model_dump_json(), corrected.model_dump_json()]

        def chat(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(message=SimpleNamespace(content=responses.pop(0)))

        monkeypatch.setattr("claims.llm.Client", lambda **kwargs: SimpleNamespace(chat=chat))
        llm = OllamaStructuredLlm("offline")
    result = StructuredAgentFinalizer(llm).finalize(finalizer_state(investigation))
    prompts = [call["messages"][-1]["content"] for call in calls] if calls else [call[0] for call in llm.calls]
    assert result == corrected and len(prompts) == 2
    assert "INCOMPATIBLE_FOLLOW_UP" in prompts[1]
    assert first.recommended_next_action == RecommendedNextAction.CONTINUE_STANDARD_REVIEW
