"""Signal codes and concrete facts must survive bounded prompt construction."""

import json
from dataclasses import replace

from agent.planner import StructuredAgentFinalizer, StructuredAgentPlanner
from agent.tools import ReadOnlyAgentTools
from agent.tools._data import MAX_SIGNALS, MAX_VALUE_CHARS
from tests.agent.review._support import prompt_data
from tests.agent.review._support import stored_review as stored_review
from tests.agent.tools._evidence import plan
from tests.infrastructure.support.agent_fakes import FakeStructuredLlm, make_recommendation, make_step


def test_stored_signal_facts_survive_planner_and_finalizer_context(stored_review):
    claim, factory, _, _ = stored_review
    observation = ReadOnlyAgentTools(factory).execute(claim.claim_id, plan("GET_INVESTIGATION"))
    item = observation.evidence[0]
    assert "ADDITIONAL_QUOTE_DAMAGE" in item.summary[:600]
    assert "HOOD" in item.summary[:600]
    assert json.loads(item.structured_data["signal_details"][0]["facts"])["damage_types"] == ["HOOD"]
    state = {
        "claim_id": claim.claim_id,
        "objective": "Review this claim",
        "max_iterations": 8,
        "iteration": 2,
        "observations": [observation.model_dump(mode="json")],
        "plan_history": [],
        "stop_reason": None,
        "evidence_registry": {item.evidence_id: item.model_dump(mode="json")},
    }
    llm = FakeStructuredLlm([make_step("FINISH"), make_recommendation([item.evidence_id])])
    StructuredAgentPlanner(llm).plan(state)
    StructuredAgentFinalizer(llm).finalize(state)
    assert "ADDITIONAL_QUOTE_DAMAGE" in prompt_data(llm.calls[0])["observations"][0]["evidence"][0]["summary"]
    details = prompt_data(llm.calls[1])["evidence"][0]["structured_data"]["signal_details"][0]
    assert details["code"] == "ADDITIONAL_QUOTE_DAMAGE"
    assert json.loads(details["facts"])["damage_types"] == ["HOOD"]
    assert details["facts_truncated"] is False
    assert factory.open_count == 0


def test_signal_details_keep_size_limits_and_report_omissions(stored_review):
    claim, factory, _, records = stored_review
    record = records["investigation"]
    signal = record.result.signals[0].model_copy(update={"evidence": {"descriptions": ["x" * 2000] * 12}})
    assessment = record.result.model_copy(update={"signals": (signal,) * (MAX_SIGNALS + 2)})
    records["investigation"] = replace(record, result=assessment)
    observation = ReadOnlyAgentTools(factory).execute(claim.claim_id, plan("GET_INVESTIGATION"))
    details = observation.data["signal_details"]
    assert len(details) == MAX_SIGNALS
    assert observation.data["signal_count"] == MAX_SIGNALS + 2
    assert observation.data["omitted_signal_count"] == 2
    assert all(len(item["facts"]) <= MAX_VALUE_CHARS and item["facts_truncated"] for item in details)
    assert len(observation.evidence[0].summary) <= 4000
    assert record.result.signals[0].evidence == {"damage_types": ["HOOD"], "count": 1}
