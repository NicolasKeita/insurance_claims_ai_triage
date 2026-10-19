"""Finalization node and its routing decision."""

from typing import Literal

from agent.graph.common import failure
from agent.models import AgentRecommendation, RecommendedNextAction, RunStatus


def finalize_node(finalizer):
    def finalize(state):
        """
        Explicit abstention, never an invented fallback answer.
        """
        try:
            if not state["evidence_registry"]:
                recommendation = AgentRecommendation(
                    case_summary="No claim evidence was collected for a supported recommendation.",
                    recommended_next_action=RecommendedNextAction.NO_RECOMMENDATION,
                    rationale=[],
                    evidence_ids=[],
                    uncertainties=["Evidence needed for a recommendation is unavailable."],
                )
            else:
                recommendation = AgentRecommendation.model_validate(finalizer.finalize(state).model_dump())
            return {"recommendation": recommendation.model_dump(mode="json")}
        except Exception:
            return failure("GENERATION")

    return finalize


def route_finalize(state) -> Literal["validate", "end"]:
    return "end" if state["status"] == RunStatus.FAILED else "validate"
