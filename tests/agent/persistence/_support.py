"""Support."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from agent.models import AgentRecommendation, AgentRunResult, EvidenceItem
from retrieval.knowledge.models import KnowledgeCitation


def make_agent_run(claim_id="CLAIM-2026-00001", *, timestamp=None, failed=False):
    timestamp = timestamp or datetime(2026, 10, 10, 10, tzinfo=timezone.utc)
    evidence = EvidenceItem(
        evidence_id=f"claim:{claim_id}",
        source_type="CLAIM_DATA",
        source_reference=claim_id,
        summary="A collision claim includes declared front bumper damage.",
        structured_data={"declared_damage": ["FRONT_BUMPER"], "amount": "3160"},
    )
    recommendation = (
        None
        if failed
        else AgentRecommendation(
            case_summary="A human should review the available claim documents.",
            recommended_next_action="REQUEST_ADDITIONAL_INFORMATION",
            rationale=[{"statement": "The claim requires document review.", "evidence_ids": [evidence.evidence_id]}],
            uncertainties=["Policy evidence has not been collected."],
            evidence_ids=[evidence.evidence_id],
        )
    )
    return _run_result(claim_id, timestamp, failed, evidence, recommendation)


def add_policy_evidence(run):
    citation = KnowledgeCitation(
        chunk_id="demo-policy-v1-collision-0",
        source_id="demo-auto-policy",
        source_version="v1",
        title="Synthetic motor policy",
        section="Collision review",
    )
    item = EvidenceItem(
        evidence_id=f"knowledge:{citation.chunk_id}",
        source_type="POLICY_CHUNK",
        source_reference=citation.chunk_id,
        summary="Check the submitted repair estimate.",
        structured_data=citation.model_dump(mode="json"),
    )
    run.evidence.append(item)
    run.recommendation.evidence_ids.append(item.evidence_id)
    run.recommendation.citations.append(citation)
    return run


def _run_result(claim_id, timestamp, failed, evidence, recommendation):
    return AgentRunResult(
        run_id=str(uuid4()),
        claim_id=claim_id,
        objective="Review the evidence and recommend a human review step.",
        status="FAILED" if failed else "COMPLETED",
        llm_model="fake-structured-model",
        recommendation=recommendation,
        tools_used=["GET_CLAIM"],
        iteration_count=1,
        tool_call_count=1,
        evidence=[evidence],
        trace=[
            {
                "iteration": 1,
                "planner_action": "GET_CLAIM",
                "planner_reason": "Need the claim context.",
                "tool_invoked": "GET_CLAIM",
                "tool_args": {},
                "observation_status": "OK",
                "evidence_ids_added": [evidence.evidence_id],
            }
        ],
        created_at=timestamp,
        completed_at=timestamp + timedelta(seconds=1),
        elapsed_seconds=1,
        error="Infrastructure tool failed." if failed else None,
    )
