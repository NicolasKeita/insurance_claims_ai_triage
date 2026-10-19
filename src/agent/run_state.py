"""Objective validation, initial graph state and run logging."""

import logging

from agent.config import DEFAULT_OBJECTIVE, MAX_OBJECTIVE_CHARS
from agent.models import RunStatus

logger = logging.getLogger(__name__)


def validated_objective(objective):
    value = DEFAULT_OBJECTIVE if objective is None else objective.strip()
    if not 1 <= len(value) <= MAX_OBJECTIVE_CHARS:
        raise ValueError(f"objective must contain 1 to {MAX_OBJECTIVE_CHARS} characters")
    return value


def initial_state(run_id, claim_id, objective, max_iterations):
    return {
        "run_id": run_id,
        "claim_id": claim_id,
        "objective": objective,
        "iteration": 0,
        "max_iterations": max_iterations,
        "plan_history": [],
        "observations": [],
        "evidence_registry": {},
        "current_plan": None,
        "invocation_counts": {},
        "trace": [],
        "recommendation": None,
        "status": RunStatus.RUNNING.value,
        "error": None,
        "failure_kind": None,
        "stop_reason": None,
        "tool_call_count": 0,
    }


def log_run(result):
    logger.info(
        "claims_agent run=%s status=%s iterations=%s tools=%s calls=%s elapsed=%.3f",
        result.run_id,
        result.status,
        result.iteration_count,
        [t.value for t in result.tools_used],
        result.tool_call_count,
        result.elapsed_seconds,
    )
