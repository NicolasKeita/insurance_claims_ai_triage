"""Run lifecycle, observability and append-only audit persistence."""

from datetime import datetime, timezone
import logging
from time import perf_counter
from uuid import uuid4

from agent.config import DEFAULT_MAX_ITERATIONS, DEFAULT_OBJECTIVE, MAX_ITERATIONS, MAX_OBJECTIVE_CHARS
from agent.graph import build_agent_graph
from agent.models import AgentRunResult, RunStatus

logger = logging.getLogger(__name__)


class AgentRunError(RuntimeError):
    def __init__(self, message: str, *, run_id: str, failure_kind: str = "TOOL"):
        super().__init__(message)
        self.run_id, self.failure_kind = run_id, failure_kind


class ClaimsAgentService:
    def __init__(self, tools, planner, finalizer, *, run_store=None,
                 max_iterations=DEFAULT_MAX_ITERATIONS, llm_model=None):
        if type(max_iterations) is not int or not 1 <= max_iterations <= MAX_ITERATIONS:
            raise ValueError(f"max_iterations must be between 1 and {MAX_ITERATIONS}")
        self.tools, self.run_store = tools, run_store
        self.max_iterations, self.llm_model = max_iterations, llm_model
        self.graph = build_agent_graph(tools, planner, finalizer)

    def review_claim(self, claim_id: str, objective: str | None = None) -> AgentRunResult:
        objective = DEFAULT_OBJECTIVE if objective is None else objective.strip()
        if not 1 <= len(objective) <= MAX_OBJECTIVE_CHARS:
            raise ValueError(f"objective must contain 1 to {MAX_OBJECTIVE_CHARS} characters")
        self.tools.require_claim(claim_id)  # Unknown claim fails before LLM/service I/O.
        started, created_at = perf_counter(), datetime.now(timezone.utc)
        run_id = str(uuid4())
        state = {"run_id": run_id, "claim_id": claim_id, "objective": objective,
                 "iteration": 0, "max_iterations": self.max_iterations, "plan_history": [],
                 "observations": [], "evidence_registry": {}, "current_plan": None,
                 "invocation_counts": {}, "trace": [], "recommendation": None,
                 "status": RunStatus.RUNNING.value, "error": None, "failure_kind": None,
                 "stop_reason": None, "tool_call_count": 0}
        try:
            state = self.graph.invoke(state, config={"recursion_limit": 2 * self.max_iterations + 6})
        except Exception:
            # Nodes preserve their own accumulated state on known failures.
            # An unexpected graph/runtime exception still needs a terminal audit
            # record; partial runtime state is unavailable from a failed invoke.
            state = {**state, "status": RunStatus.FAILED.value, "recommendation": None,
                     "error": "Agent orchestration failed.", "failure_kind": "ORCHESTRATION"}
        tools_used = list(dict.fromkeys(t["tool_invoked"] for t in state["trace"] if t["tool_invoked"]))
        result = AgentRunResult(
            run_id=run_id, claim_id=claim_id, objective=objective, status=state["status"],
            llm_model=self.llm_model, recommendation=state["recommendation"], tools_used=tools_used,
            iteration_count=state["iteration"], tool_call_count=state["tool_call_count"],
            evidence=list(state["evidence_registry"].values()), trace=state["trace"],
            created_at=created_at, completed_at=datetime.now(timezone.utc),
            elapsed_seconds=perf_counter() - started, error=state["error"])
        if self.run_store is not None:
            self.run_store.save(result)
        logger.info("claims_agent run=%s status=%s iterations=%s tools=%s calls=%s elapsed=%.3f",
                    run_id, result.status, result.iteration_count,
                    [t.value for t in result.tools_used], result.tool_call_count, result.elapsed_seconds)
        if result.status == RunStatus.FAILED:
            raise AgentRunError(result.error, run_id=run_id, failure_kind=state["failure_kind"])
        return result
