"""Run lifecycle, observability and append-only audit persistence."""

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from agent.config import DEFAULT_MAX_ITERATIONS, MAX_ITERATIONS
from agent.graph import build_agent_graph
from agent.models import AgentRunResult, RunStatus
from agent.run_state import initial_state, log_run, validated_objective


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


class ClaimsAgentService:
    def __init__(self, config: AgentServiceConfig):
        if type(config.max_iterations) is not int or not 1 <= config.max_iterations <= MAX_ITERATIONS:
            raise ValueError(f"max_iterations must be between 1 and {MAX_ITERATIONS}")
        self.tools, self.run_store = config.tools, config.run_store
        self.max_iterations, self.llm_model = config.max_iterations, config.llm_model
        self.graph = build_agent_graph(config.tools, config.planner, config.finalizer)

    def review_claim(self, claim_id: str, objective: str | None = None) -> AgentRunResult:
        """
        Unknown claim fails before LLM/service I/O.

        Nodes preserve their own accumulated state on known failures. An unexpected graph/runtime
        exception still needs a terminal audit record; partial runtime state is unavailable from a
        failed invoke.
        """
        objective = validated_objective(objective)
        self.tools.require_claim(claim_id)
        started, created_at = perf_counter(), datetime.now(timezone.utc)
        run_id = str(uuid4())
        state = initial_state(run_id, claim_id, objective, self.max_iterations)
        state = self._invoke_graph(state)
        result = self._build_result(state, run_id, created_at, started)
        if self.run_store is not None:
            self.run_store.save(result)
        log_run(result)
        if result.status == RunStatus.FAILED:
            raise AgentRunError(result.error, run_id=run_id, failure_kind=state["failure_kind"])
        return result

    def _invoke_graph(self, state):
        try:
            return self.graph.invoke(state, config={"recursion_limit": 2 * self.max_iterations + 6})
        except Exception:
            return {
                **state,
                "status": RunStatus.FAILED.value,
                "recommendation": None,
                "error": "Agent orchestration failed.",
                "failure_kind": "ORCHESTRATION",
            }

    def _build_result(self, state, run_id, created_at, started):
        tools_used = list(dict.fromkeys(t["tool_invoked"] for t in state["trace"] if t["tool_invoked"]))
        return AgentRunResult(
            run_id=run_id,
            claim_id=state["claim_id"],
            objective=state["objective"],
            status=state["status"],
            llm_model=self.llm_model,
            recommendation=state["recommendation"],
            tools_used=tools_used,
            iteration_count=state["iteration"],
            tool_call_count=state["tool_call_count"],
            evidence=list(state["evidence_registry"].values()),
            trace=state["trace"],
            created_at=created_at,
            completed_at=datetime.now(timezone.utc),
            elapsed_seconds=perf_counter() - started,
            error=state["error"],
        )
