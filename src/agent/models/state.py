"""Mutable graph state shared by the agent nodes."""

from typing import TypedDict


class AgentState(TypedDict):
    run_id: str
    claim_id: str
    objective: str
    iteration: int
    max_iterations: int
    plan_history: list[dict]
    observations: list[dict]
    evidence_registry: dict[str, dict]
    current_plan: dict | None
    invocation_counts: dict[str, int]
    trace: list[dict]
    recommendation: dict | None
    status: str
    error: str | None
    failure_kind: str | None
    stop_reason: str | None
    tool_call_count: int
