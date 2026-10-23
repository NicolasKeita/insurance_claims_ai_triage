from typing import Literal

from agent.evidence import (
    EvidenceValidationError,
    rationale_evidence_ids,
    validate_recommendation,
    validate_recommendation_versions,
)
from agent.graph._failure import failure
from agent.models import (
    AgentRecommendation,
    EvidenceItem,
    ObservationStatus,
    RecommendedNextAction,
    RunStatus,
)
from agent.review_policy import visual_uncertainties


def finalize(state, finalizer):
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
    except EvidenceValidationError as error:
        return failure("GROUNDING", code=error.code)
    except Exception:
        return failure("GENERATION")


def route_finalize(state) -> Literal["validate", "end"]:
    return "end" if state["status"] == RunStatus.FAILED else "validate"


def validate(state):
    """
    Application-observed limits must survive a finalizer filling all available uncertainty slots
    with its own observations.
    """
    try:
        recommendation = AgentRecommendation.model_validate(state["recommendation"])
        validate_recommendation_versions(recommendation)
        registry = {k: EvidenceItem.model_validate(v) for k, v in state["evidence_registry"].items()}
        citations = validate_recommendation(recommendation, registry, observations=state["observations"])
        recommendation.citations = citations
        recommendation.evidence_ids = rationale_evidence_ids(recommendation)
        limitations = [
            f"{o['tool_name']}: {o.get('message') or o['status']}"
            for o in state["observations"]
            if o["status"] in (ObservationStatus.NOT_AVAILABLE, ObservationStatus.NOT_FOUND)
        ]
        if state["stop_reason"] == "Maximum tool steps reached.":
            limitations.append(state["stop_reason"])
        limitations.extend(visual_uncertainties(registry))
        recommendation.uncertainties = list(dict.fromkeys([*limitations, *recommendation.uncertainties]))[:20]
        recommendation = AgentRecommendation.model_validate(recommendation.model_dump())
        return {
            "recommendation": recommendation.model_dump(mode="json"),
            "status": RunStatus.COMPLETED.value,
        }
    except EvidenceValidationError as error:
        return failure("GROUNDING", code=error.code)
    except Exception:
        return failure("GENERATION")
