"""The restricted enum must reach the actual Ollama request, without network I/O."""

from types import SimpleNamespace

from agent.models import RecommendedNextAction
from agent.planner import StructuredAgentFinalizer
from claims.llm import OllamaStructuredLlm
from tests.agent.review._support import finalizer_state
from tests.agent.review._support import review_evidence as review_evidence
from tests.agent.review._support import stored_review as stored_review
from tests.infrastructure.support.agent_fakes import make_recommendation


def test_repair_follow_up_enum_is_sent_in_ollama_format(monkeypatch, review_evidence):
    calls = []
    recommendation = make_recommendation([review_evidence.evidence_id])
    recommendation.recommended_next_action = RecommendedNextAction.REFER_TO_EXPERT

    class RecordingClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(message=SimpleNamespace(content=recommendation.model_dump_json()))

    monkeypatch.setattr("claims.llm.Client", RecordingClient)
    llm = OllamaStructuredLlm("fake-local", context_window=8192)
    result = StructuredAgentFinalizer(llm).finalize(finalizer_state(review_evidence))
    actions = calls[0]["format"]["properties"]["recommended_next_action"]["enum"]
    assert set(actions) == {"REQUEST_ADDITIONAL_INFORMATION", "REFER_TO_EXPERT", "REFER_FOR_INVESTIGATION_REVIEW"}
    assert result.recommended_next_action == RecommendedNextAction.REFER_TO_EXPERT
    assert calls[0]["options"] == {"temperature": 0, "seed": 42, "num_ctx": 8192}
    assert len(calls) == 1
