"""Agent run result <-> audit row conversion with JSON snapshotting."""

import json
from uuid import UUID

from agent.models import AgentRunResult
from persistence.models import AgentRunRow


def agent_run_to_row(claim_pk: UUID, result: AgentRunResult) -> AgentRunRow:
    """
    Validate again and snapshot JSON values before issuing an INSERT.

    Pydantic's JSON mode can normalize NaN to null; reject the original nonfinite values before that conversion
    loses their meaning.

    PostgreSQL JSONB cannot represent nonfinite numbers. Also detach all nested snapshots from mutable
    caller-owned Pydantic lists/dictionaries.
    """
    checked = AgentRunResult.model_validate(result.model_dump())
    json.dumps(checked.model_dump(), default=str, allow_nan=False)
    payload = checked.model_dump(mode="json")
    payload = json.loads(json.dumps(payload, allow_nan=False))
    return AgentRunRow(
        claim_pk=claim_pk,
        run_id=payload["run_id"],
        objective=payload["objective"],
        status=payload["status"],
        agent_version=payload["agent_version"],
        prompt_version=payload["prompt_version"],
        llm_model=payload["llm_model"],
        iteration_count=payload["iteration_count"],
        tool_call_count=payload["tool_call_count"],
        tools_used=payload["tools_used"],
        evidence=payload["evidence"],
        trace=payload["trace"],
        recommendation=payload["recommendation"],
        created_at=checked.created_at,
        completed_at=checked.completed_at,
        elapsed_seconds=payload["elapsed_seconds"],
        error=payload["error"],
    )


def agent_run_from_row(row: AgentRunRow, claim_id: str) -> AgentRunResult:
    return AgentRunResult.model_validate(
        {
            "run_id": row.run_id,
            "claim_id": claim_id,
            "objective": row.objective,
            "status": row.status,
            "agent_version": row.agent_version,
            "prompt_version": row.prompt_version,
            "llm_model": row.llm_model,
            "iteration_count": row.iteration_count,
            "tool_call_count": row.tool_call_count,
            "tools_used": row.tools_used,
            "evidence": row.evidence,
            "trace": row.trace,
            "recommendation": row.recommendation,
            "created_at": row.created_at,
            "completed_at": row.completed_at,
            "elapsed_seconds": row.elapsed_seconds,
            "error": row.error,
        }
    )
