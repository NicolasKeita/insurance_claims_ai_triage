import json
from typing import Literal

from agent.evidence import EvidenceValidationError
from agent.graph._failure import failure
from agent.models import (
    AgentPlanStep,
    AgentTraceStep,
    ObservationStatus,
    RunStatus,
    ToolObservation,
)


def tool(state, tools):
    step = AgentPlanStep.model_validate(state["current_plan"])
    args = step.validated_arguments().model_dump(mode="json")
    signature = json.dumps([step.action.value, args], sort_keys=True)
    counts = dict(state["invocation_counts"])
    duplicate = counts.get(signature, 0) >= 1
    invoked = None if duplicate else step.tool_name
    counts[signature] = counts.get(signature, 0) + 1
    call_count = state["tool_call_count"] + (0 if duplicate else 1)
    trace = AgentTraceStep(
        iteration=state["iteration"],
        planner_action=step.action,
        planner_reason=step.reason,
        tool_invoked=invoked,
        tool_args=args,
    )
    try:
        observation = invoke_tool(state, tools, step, duplicate)
        registry, added = register_evidence(state, observation)
        trace.observation_status = observation.status
        trace.evidence_ids_added = added
        return {
            "observations": [
                *state["observations"],
                observation.model_dump(mode="json"),
            ],
            "evidence_registry": registry,
            "invocation_counts": counts,
            "tool_call_count": call_count,
            "trace": [*state["trace"], trace.model_dump(mode="json")],
        }
    except Exception:
        return {
            **failure("TOOL"),
            "invocation_counts": counts,
            "tool_call_count": call_count,
            "trace": [*state["trace"], trace.model_dump(mode="json")],
        }


def route_tool(state) -> Literal["plan", "end"]:
    return "end" if state["status"] == RunStatus.FAILED else "plan"


def register_evidence(state, observation):
    registry = dict(state["evidence_registry"])
    added = []
    for item in observation.evidence:
        payload = item.model_dump(mode="json")
        if item.evidence_id in registry and registry[item.evidence_id] != payload:
            raise EvidenceValidationError("Conflicting evidence shares one ID")
        if item.evidence_id not in registry:
            registry[item.evidence_id] = payload
            added.append(item.evidence_id)
    return registry, added


def invoke_tool(state, tools, step, duplicate):
    if duplicate:
        observation = ToolObservation(
            tool_name=step.tool_name,
            status=ObservationStatus.DUPLICATE_BLOCKED,
            message="Exact read-only call already attempted. Choose another action or FINISH.",
        )
    else:
        observation = ToolObservation.model_validate(tools.execute(state["claim_id"], step).model_dump())
        if observation.tool_name != step.tool_name:
            raise ValueError("Tool returned an observation for a different action")
    return observation
