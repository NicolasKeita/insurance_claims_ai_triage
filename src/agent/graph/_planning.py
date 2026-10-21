from typing import Literal

from agent.graph._failure import failure
from agent.models import (
    AgentAction,
    AgentPlanStep,
    AgentTraceStep,
    RunStatus,
)


def plan(state, planner):
    if state["iteration"] >= state["max_iterations"]:
        return {"current_plan": None, "stop_reason": "Maximum tool steps reached."}
    try:
        step = AgentPlanStep.model_validate(planner.plan(state).model_dump())
    except Exception:
        return failure("GENERATION")
    update = {
        "current_plan": step.model_dump(mode="json"),
        "plan_history": [*state["plan_history"], step.model_dump(mode="json")],
    }
    if step.action == AgentAction.FINISH:
        trace = AgentTraceStep(
            iteration=state["iteration"] + 1,
            planner_action=step.action,
            planner_reason=step.reason,
        )
        update.update(
            stop_reason="Planner finished evidence collection.",
            trace=[*state["trace"], trace.model_dump(mode="json")],
        )
    else:
        update["iteration"] = state["iteration"] + 1
    return update


def route_plan(state) -> Literal["tool", "finalize", "end"]:
    if state["status"] == RunStatus.FAILED:
        return "end"
    plan = state["current_plan"]
    return "finalize" if plan is None or plan["action"] == AgentAction.FINISH else "tool"
