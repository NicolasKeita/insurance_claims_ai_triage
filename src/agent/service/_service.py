"""Service."""

from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from agent.config import DEFAULT_OBJECTIVE, MAX_ITERATIONS, MAX_OBJECTIVE_CHARS
from agent.graph import build_agent_graph
from agent.models import AgentRunResult, RunStatus
from agent.service._constants import logger
from agent.service._contracts import AgentRunError, AgentServiceConfig


class ClaimsAgentService:
    def __init__(self, config: AgentServiceConfig):
        if type(config.max_iterations) is not int or not 1 <= config.max_iterations <= MAX_ITERATIONS:
            raise ValueError(f"max_iterations must be between 1 and {MAX_ITERATIONS}")
        if config.max_iterations < getattr(config.planner, "minimum_tool_steps", 1):
            raise ValueError("max_iterations must allow all required initial reads")
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
        objective = DEFAULT_OBJECTIVE if objective is None else objective.strip()
        if not 1 <= len(objective) <= MAX_OBJECTIVE_CHARS:
            raise ValueError(f"objective must contain 1 to {MAX_OBJECTIVE_CHARS} characters")
        self.tools.require_claim(claim_id)
        started, created_at = perf_counter(), datetime.now(timezone.utc)
        run_id = str(uuid4())
        state = self._initial_state(run_id, claim_id, objective)
        try:
            state = self.graph.invoke(state, config={"recursion_limit": 2 * self.max_iterations + 6})
        except Exception:
            state = {
                **state,
                "status": RunStatus.FAILED.value,
                "recommendation": None,
                "error": "Agent orchestration failed.",
                "failure_kind": "ORCHESTRATION",
            }
        result = self._result(state, created_at, started)
        if self.run_store is not None:
            self.run_store.save(result)
        self._log_result(run_id, result)
        if result.status == RunStatus.FAILED:
            raise AgentRunError(result.error, run_id=run_id, failure_kind=state["failure_kind"])
        return result

    def _initial_state(self, run_id, claim_id, objective):
        return {
            "run_id": run_id,
            "claim_id": claim_id,
            "objective": objective,
            "iteration": 0,
            "max_iterations": self.max_iterations,
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

    def _result(self, state, created_at, started):
        tools_used = list(dict.fromkeys(t["tool_invoked"] for t in state["trace"] if t["tool_invoked"]))
        return AgentRunResult(
            run_id=state["run_id"],
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

    def _log_result(self, run_id, result):
        logger.info(
            "claims_agent run=%s status=%s iterations=%s tools=%s calls=%s elapsed=%.3f",
            run_id,
            result.status,
            result.iteration_count,
            [t.value for t in result.tools_used],
            result.tool_call_count,
            result.elapsed_seconds,
        )
