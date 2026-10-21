"""Test context."""

import json
from types import SimpleNamespace

from agent.config import MAX_FINALIZER_EVIDENCE_CHARS
from agent.planner import StructuredAgentFinalizer
from claims.llm import OllamaStructuredLlm
from tests.agent.safety._constants import CLAIM_ID
from tests.infrastructure.support.agent_fakes import (
    FakeStructuredLlm,
    make_evidence,
    make_recommendation,
    make_step,
)


def test_large_finalizer_evidence_is_bounded_without_losing_ids():
    llm = FakeStructuredLlm([make_recommendation()])
    registry = {}
    for index in range(40):
        item = make_evidence(f"claim:context-{index}", summary='\\"' * 1900)
        item.structured_data = {"free_text": "x" * 12000}
        registry[item.evidence_id] = item.model_dump(mode="json")
    output = StructuredAgentFinalizer(llm).finalize(
        {
            "claim_id": CLAIM_ID,
            "objective": "Review evidence",
            "stop_reason": "Maximum tool steps reached.",
            "evidence_registry": registry,
            "observations": [],
        }
    )
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
