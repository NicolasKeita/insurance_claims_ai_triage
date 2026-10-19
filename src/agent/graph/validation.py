"""Final grounding validation of the recommendation against collected evidence."""

from agent.config import CLAIMS_AGENT_PROMPT_VERSION, CLAIMS_AGENT_VERSION
from agent.evidence import EvidenceValidationError, validate_recommendation
from agent.graph.common import failure
from agent.models import AgentRecommendation, EvidenceItem, ObservationStatus, RunStatus


def validate_node(state):
    """
    Application-observed limits must survive a finalizer filling all available uncertainty slots
    with its own observations.
    """
    try:
        recommendation = _validated_recommendation(state)
        return {"recommendation": recommendation.model_dump(mode="json"), "status": RunStatus.COMPLETED.value}
    except EvidenceValidationError:
        return failure("GROUNDING")
    except Exception:
        return failure("GENERATION")


def _validated_recommendation(state):
    recommendation = AgentRecommendation.model_validate(state["recommendation"])
    _check_versions(recommendation)
    registry = {k: EvidenceItem.model_validate(v) for k, v in state["evidence_registry"].items()}
    recommendation.citations = validate_recommendation(recommendation, registry)
    recommendation.evidence_ids = list(
        dict.fromkeys(
            [
                *recommendation.evidence_ids,
                *(e for rationale in recommendation.rationale for e in rationale.evidence_ids),
            ]
        )
    )
    limitations = _limitations(state)
    recommendation.uncertainties = list(dict.fromkeys([*limitations, *recommendation.uncertainties]))[:20]
    return AgentRecommendation.model_validate(recommendation.model_dump())


def _check_versions(recommendation):
    if (
        recommendation.agent_version != CLAIMS_AGENT_VERSION
        or recommendation.prompt_version != CLAIMS_AGENT_PROMPT_VERSION
    ):
        raise EvidenceValidationError("Recommendation versions do not match this agent")


def _limitations(state):
    limitations = [
        f"{o['tool_name']}: {o.get('message') or o['status']}"
        for o in state["observations"]
        if o["status"] in (ObservationStatus.NOT_AVAILABLE, ObservationStatus.NOT_FOUND)
    ]
    if state["stop_reason"] == "Maximum tool steps reached.":
        limitations.append(state["stop_reason"])
    return limitations
