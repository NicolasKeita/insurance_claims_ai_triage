"""Agent audit mapping remains independent of PostgreSQL and inference servers."""

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agent.models import AgentRecommendation, AgentRunResult, EvidenceItem
from knowledge.models import KnowledgeCitation
from persistence.agent_runs import AgentRunRepository, agent_run_from_row, agent_run_to_row
from persistence.models import AgentRunRow


def test_historical_versions_remain_readable_after_current_version_changes():
    run = make_agent_run()
    run.agent_version = run.recommendation.agent_version = "claims_agent_prior_version"
    run.prompt_version = run.recommendation.prompt_version = "claims_agent_prompt_prior_version"
    row = agent_run_to_row(uuid4(), run)
    assert agent_run_from_row(row, run.claim_id) == run


def make_agent_run(claim_id="CLAIM-2026-00001", *, timestamp=None, failed=False):
    timestamp = timestamp or datetime(2026, 10, 10, 10, tzinfo=timezone.utc)
    evidence = EvidenceItem(
        evidence_id=f"claim:{claim_id}", source_type="CLAIM_DATA", source_reference=claim_id,
        summary="A collision claim includes declared front bumper damage.",
        structured_data={"declared_damage": ["FRONT_BUMPER"], "amount": "3160"},
    )
    recommendation = None if failed else AgentRecommendation(
        case_summary="A human should review the available claim documents.",
        recommended_next_action="REQUEST_ADDITIONAL_INFORMATION",
        rationale=[{"statement": "The claim requires document review.",
                    "evidence_ids": [evidence.evidence_id]}],
        uncertainties=["Policy evidence has not been collected."],
        evidence_ids=[evidence.evidence_id],
    )
    return AgentRunResult(
        run_id=str(uuid4()), claim_id=claim_id,
        objective="Review the evidence and recommend a human review step.",
        status="FAILED" if failed else "COMPLETED", llm_model="fake-structured-model",
        recommendation=recommendation, tools_used=["GET_CLAIM"], iteration_count=1,
        tool_call_count=1, evidence=[evidence],
        trace=[{"iteration": 1, "planner_action": "GET_CLAIM",
                "planner_reason": "Need the claim context.", "tool_invoked": "GET_CLAIM",
                "tool_args": {}, "observation_status": "OK",
                "evidence_ids_added": [evidence.evidence_id]}],
        created_at=timestamp, completed_at=timestamp + timedelta(seconds=1),
        elapsed_seconds=1, error="Infrastructure tool failed." if failed else None,
    )


def add_policy_evidence(run):
    citation = KnowledgeCitation(
        chunk_id="demo-policy-v1-collision-0", source_id="demo-auto-policy",
        source_version="v1", title="Synthetic motor policy", section="Collision review",
    )
    item = EvidenceItem(
        evidence_id=f"knowledge:{citation.chunk_id}", source_type="POLICY_CHUNK",
        source_reference=citation.chunk_id, summary="Check the submitted repair estimate.",
        structured_data=citation.model_dump(mode="json"),
    )
    run.evidence.append(item)
    run.recommendation.evidence_ids.append(item.evidence_id)
    run.recommendation.citations.append(citation)
    return run


@pytest.mark.parametrize("failed", [False, True])
def test_agent_run_mapping_round_trip_preserves_audit_and_terminal_status(failed):
    source = make_agent_run(failed=failed)
    pk = uuid4()
    row = agent_run_to_row(pk, source)
    assert row.claim_pk == pk
    assert row.run_id == source.run_id
    assert row.status == ("FAILED" if failed else "COMPLETED")
    assert row.recommendation is None if failed else row.recommendation["human_review_required"] is True
    assert agent_run_from_row(row, source.claim_id) == source
    json.dumps([row.tools_used, row.evidence, row.trace, row.recommendation], allow_nan=False)


def test_agent_run_mapping_detaches_json_snapshots():
    source = make_agent_run()
    row = agent_run_to_row(uuid4(), source)
    source.evidence[0].structured_data["declared_damage"].append("REAR_BUMPER")
    source.trace[0].tool_args["changed_later"] = True
    source.recommendation.uncertainties.append("Added after snapshot.")
    assert row.evidence[0]["structured_data"]["declared_damage"] == ["FRONT_BUMPER"]
    assert row.trace[0]["tool_args"] == {}
    assert row.recommendation["uncertainties"] == ["Policy evidence has not been collected."]


def test_agent_audit_mapping_preserves_policy_citation_metadata():
    source = add_policy_evidence(make_agent_run())
    row = agent_run_to_row(uuid4(), source)
    loaded = agent_run_from_row(row, source.claim_id)
    assert loaded == source
    citation = loaded.recommendation.citations[0]
    assert (citation.chunk_id, citation.source_id, citation.source_version, citation.section) == (
        "demo-policy-v1-collision-0", "demo-auto-policy", "v1", "Collision review",
    )
    assert loaded.evidence[-1].structured_data == citation.model_dump(mode="json")


def test_agent_mapper_revalidates_unsafe_model_copy():
    source = make_agent_run().model_copy(update={"recommendation": None})
    with pytest.raises(ValidationError, match="completed run"):
        agent_run_to_row(uuid4(), source)


def test_agent_mapper_rejects_nonfinite_json_evidence():
    source = make_agent_run()
    source.evidence[0].structured_data["score"] = float("nan")
    with pytest.raises(ValueError, match="JSON compliant"):
        agent_run_to_row(uuid4(), source)


@pytest.mark.parametrize("limit", [0, -1, 101, True, 2.5, "20"])
def test_agent_run_list_limit_is_bounded_before_database_access(limit):
    repository = AgentRunRepository(None)
    with pytest.raises(ValueError, match="between 1 and 100"):
        repository.list_for_claim("claim", limit)


def test_agent_audit_row_has_restricting_fk_and_no_update_delete_api():
    assert next(iter(AgentRunRow.__table__.c.claim_pk.foreign_keys)).ondelete is None
    assert not hasattr(AgentRunRepository, "update")
    assert not hasattr(AgentRunRepository, "delete")
    assert AgentRunRow.__table__.c.insertion_order.identity.always is True
