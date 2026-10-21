"""Contracts."""

from dataclasses import dataclass

from agent.config import (
    DEFAULT_MAX_ITERATIONS,
)


@dataclass
class AgentServiceConfig:
    """Dependencies and execution limits for claim reviews."""

    tools: object
    planner: object
    finalizer: object
    run_store: object = None
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    llm_model: object = None


class AgentRunError(RuntimeError):
    def __init__(self, message: str, *, run_id: str, failure_kind: str = "TOOL"):
        super().__init__(message)
        self.run_id, self.failure_kind = run_id, failure_kind
