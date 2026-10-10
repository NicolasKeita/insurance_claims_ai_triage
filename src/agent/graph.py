"""Inspectable PLAN / TOOL / FINALIZE / VALIDATE graph with bounded execution."""

import json
from typing import Literal

from langgraph.graph import END, START, StateGraph

from agent.config import CLAIMS_AGENT_PROMPT_VERSION, CLAIMS_AGENT_VERSION
from agent.evidence import EvidenceValidationError, validate_recommendation
from agent.models import (
    AgentAction, AgentPlanStep, AgentRecommendation, AgentState, AgentTraceStep,
    EvidenceItem, ObservationStatus, RecommendedNextAction, RunStatus, ToolObservation,
)


def failure(kind: str) -> dict:
    # Never persist raw exceptions, prompts, credentials or provider responses.
    return {"status": RunStatus.FAILED.value, "recommendation": None,
            "error": f"Agent {kind.lower()} failed.",
            "failure_kind": kind}


def build_agent_graph(tools, planner, finalizer):
    def plan(state):
        if state["iteration"] >= state["max_iterations"]:
            return {"current_plan": None, "stop_reason": "Maximum tool steps reached."}
        try:
            step = AgentPlanStep.model_validate(planner.plan(state).model_dump())
        except Exception:
            return failure("GENERATION")
        update = {"current_plan": step.model_dump(mode="json"),
                  "plan_history": [*state["plan_history"], step.model_dump(mode="json")]}
        if step.action == AgentAction.FINISH:
            trace = AgentTraceStep(iteration=state["iteration"] + 1,
                                   planner_action=step.action, planner_reason=step.reason)
            update.update(stop_reason="Planner finished evidence collection.",
                          trace=[*state["trace"], trace.model_dump(mode="json")])
        else:
            update["iteration"] = state["iteration"] + 1
        return update

    def route_plan(state) -> Literal["tool", "finalize", "end"]:
        if state["status"] == RunStatus.FAILED:
            return "end"
        plan = state["current_plan"]
        return "finalize" if plan is None or plan["action"] == AgentAction.FINISH else "tool"

    def tool(state):
        step = AgentPlanStep.model_validate(state["current_plan"])
        args = step.validated_arguments().model_dump(mode="json")
        signature = json.dumps([step.action.value, args], sort_keys=True)
        counts = dict(state["invocation_counts"])
        duplicate = counts.get(signature, 0) >= 1
        invoked = None if duplicate else step.tool_name
        counts[signature] = counts.get(signature, 0) + 1
        call_count = state["tool_call_count"] + (0 if duplicate else 1)
        trace = AgentTraceStep(iteration=state["iteration"], planner_action=step.action,
                               planner_reason=step.reason, tool_invoked=invoked, tool_args=args)
        try:
            if duplicate:
                observation = ToolObservation(tool_name=step.tool_name,
                    status=ObservationStatus.DUPLICATE_BLOCKED,
                    message="Exact read-only call already attempted. Choose another action or FINISH.")
            else:
                observation = ToolObservation.model_validate(
                    tools.execute(state["claim_id"], step).model_dump())
                if observation.tool_name != step.tool_name:
                    raise ValueError("Tool returned an observation for a different action")
            registry = dict(state["evidence_registry"])
            added = []
            for item in observation.evidence:
                payload = item.model_dump(mode="json")
                if item.evidence_id in registry and registry[item.evidence_id] != payload:
                    raise EvidenceValidationError("Conflicting evidence shares one ID")
                if item.evidence_id not in registry:
                    registry[item.evidence_id] = payload
                    added.append(item.evidence_id)
            trace.observation_status = observation.status
            trace.evidence_ids_added = added
            return {"observations": [*state["observations"], observation.model_dump(mode="json")],
                    "evidence_registry": registry, "invocation_counts": counts,
                    "tool_call_count": call_count,
                    "trace": [*state["trace"], trace.model_dump(mode="json")]}
        except Exception:
            return {**failure("TOOL"), "invocation_counts": counts,
                    "tool_call_count": call_count,
                    "trace": [*state["trace"], trace.model_dump(mode="json")]}

    def route_tool(state) -> Literal["plan", "end"]:
        return "end" if state["status"] == RunStatus.FAILED else "plan"

    def finalize(state):
        try:
            if not state["evidence_registry"]:
                # Explicit abstention, never an invented fallback answer.
                recommendation = AgentRecommendation(
                    case_summary="No claim evidence was collected for a supported recommendation.",
                    recommended_next_action=RecommendedNextAction.NO_RECOMMENDATION,
                    rationale=[], evidence_ids=[],
                    uncertainties=["Evidence needed for a recommendation is unavailable."])
            else:
                recommendation = AgentRecommendation.model_validate(finalizer.finalize(state).model_dump())
            return {"recommendation": recommendation.model_dump(mode="json")}
        except Exception:
            return failure("GENERATION")

    def route_finalize(state) -> Literal["validate", "end"]:
        return "end" if state["status"] == RunStatus.FAILED else "validate"

    def validate(state):
        try:
            recommendation = AgentRecommendation.model_validate(state["recommendation"])
            if (recommendation.agent_version != CLAIMS_AGENT_VERSION
                    or recommendation.prompt_version != CLAIMS_AGENT_PROMPT_VERSION):
                raise EvidenceValidationError("Recommendation versions do not match this agent")
            registry = {k: EvidenceItem.model_validate(v) for k, v in state["evidence_registry"].items()}
            citations = validate_recommendation(recommendation, registry)
            recommendation.citations = citations
            recommendation.evidence_ids = list(dict.fromkeys([
                *recommendation.evidence_ids,
                *(e for rationale in recommendation.rationale for e in rationale.evidence_ids),
            ]))
            limitations = [f"{o['tool_name']}: {o.get('message') or o['status']}"
                           for o in state["observations"]
                           if o["status"] in (ObservationStatus.NOT_AVAILABLE, ObservationStatus.NOT_FOUND)]
            if state["stop_reason"] == "Maximum tool steps reached.":
                limitations.append(state["stop_reason"])
            # Application-observed limits must survive a finalizer filling all
            # available uncertainty slots with its own observations.
            recommendation.uncertainties = list(dict.fromkeys(
                [*limitations, *recommendation.uncertainties]))[:20]
            recommendation = AgentRecommendation.model_validate(recommendation.model_dump())
            return {"recommendation": recommendation.model_dump(mode="json"),
                    "status": RunStatus.COMPLETED.value}
        except EvidenceValidationError:
            return failure("GROUNDING")
        except Exception:
            return failure("GENERATION")

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("tool", tool)
    graph.add_node("finalize", finalize)
    graph.add_node("validate", validate)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_plan, {"tool": "tool", "finalize": "finalize", "end": END})
    graph.add_conditional_edges("tool", route_tool, {"plan": "plan", "end": END})
    graph.add_conditional_edges("finalize", route_finalize, {"validate": "validate", "end": END})
    graph.add_edge("validate", END)
    return graph.compile()
