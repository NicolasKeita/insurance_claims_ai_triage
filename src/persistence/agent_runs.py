"""Append-only agent audit snapshots, separate from claim/assessment writes."""

import json
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from agent.models import AgentRunResult
from persistence.database import transaction
from persistence.models import AgentRunRow, ClaimRow
from persistence.repositories import ClaimRepository


class AgentRunAlreadyExistsError(ValueError):
    pass


def agent_run_to_row(claim_pk: UUID, result: AgentRunResult) -> AgentRunRow:
    """Validate again and snapshot JSON values before issuing an INSERT."""
    checked = AgentRunResult.model_validate(result.model_dump())
    # Pydantic's JSON mode can normalize NaN to null; reject the original
    # nonfinite values before that conversion loses their meaning.
    json.dumps(checked.model_dump(), default=str, allow_nan=False)
    payload = checked.model_dump(mode="json")
    # PostgreSQL JSONB cannot represent nonfinite numbers. Also detach all
    # nested snapshots from mutable caller-owned Pydantic lists/dictionaries.
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
    return AgentRunResult.model_validate({
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
    })


class AgentRunRepository:
    """Only INSERT and SELECT; flushes without committing its unit of work."""

    def __init__(self, session: Session):
        self.session = session
        self.claims = ClaimRepository(session)

    def add(self, result: AgentRunResult) -> AgentRunResult:
        row = agent_run_to_row(self.claims.require_pk(result.claim_id), result)
        self.session.add(row)
        try:
            self.session.flush()
        except IntegrityError as error:
            name = getattr(getattr(error.orig, "diag", None), "constraint_name", None)
            if name == "uq_agent_runs_run_id":
                raise AgentRunAlreadyExistsError(f"Agent run {result.run_id} already exists") from error
            raise
        return agent_run_from_row(row, result.claim_id)

    def get_by_run_id(self, run_id: str) -> AgentRunResult | None:
        record = self.session.execute(
            select(AgentRunRow, ClaimRow.claim_id)
            .join(ClaimRow, AgentRunRow.claim_pk == ClaimRow.id)
            .where(AgentRunRow.run_id == run_id)
        ).first()
        return agent_run_from_row(record[0], record[1]) if record else None

    def list_for_claim(self, claim_id: str, limit: int = 20) -> list[AgentRunResult]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("Agent run list limit must be between 1 and 100")
        pk = self.claims.require_pk(claim_id)
        rows = self.session.scalars(
            select(AgentRunRow).where(AgentRunRow.claim_pk == pk)
            .order_by(AgentRunRow.created_at.desc(), AgentRunRow.insertion_order.desc())
            .limit(limit)
        )
        return [agent_run_from_row(row, claim_id) for row in rows]

    def latest_for_claim(self, claim_id: str) -> AgentRunResult | None:
        results = self.list_for_claim(claim_id, limit=1)
        return results[0] if results else None


class AgentRunStore:
    """Application service owns transactions and returns detached domain models."""

    def __init__(self, factory: sessionmaker[Session]):
        self.factory = factory

    def save(self, result: AgentRunResult) -> AgentRunResult:
        with transaction(self.factory) as session:
            return AgentRunRepository(session).add(result)

    def get_by_run_id(self, run_id: str) -> AgentRunResult | None:
        with self.factory() as session:
            return AgentRunRepository(session).get_by_run_id(run_id)

    def list_for_claim(self, claim_id: str, limit: int = 20) -> list[AgentRunResult]:
        with self.factory() as session:
            return AgentRunRepository(session).list_for_claim(claim_id, limit)

    def latest_for_claim(self, claim_id: str) -> AgentRunResult | None:
        with self.factory() as session:
            return AgentRunRepository(session).latest_for_claim(claim_id)
