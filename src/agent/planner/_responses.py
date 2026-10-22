"""Select the response schema from application-owned collected evidence."""

from typing import Literal

from pydantic import Field, create_model

from agent.config import CLAIMS_AGENT_PROMPT_VERSION, CLAIMS_AGENT_VERSION
from agent.models import AgentRecommendation, RecommendationRationale, RecommendedNextAction
from agent.review_policy import REPAIR_FOLLOW_UP_ACTIONS, RepairFollowUpAction, repair_discrepancy_sources


class RepairFollowUpRecommendation(AgentRecommendation):
    recommended_next_action: RepairFollowUpAction


def _response_model(registry, sources):
    """Constrain this generation without changing historical persistence contracts.

    An abstention may have no rationale; each supplied rationale must still cite
    collected evidence. A required repair follow-up cannot be an abstention.
    """
    evidence_id = Literal[tuple(registry)] if registry else str
    references = Field(min_length=1 if registry else 0, max_length=40 if registry else 0)
    rationale = create_model(
        "CollectedEvidenceRationale",
        __base__=RecommendationRationale,
        evidence_ids=(list[evidence_id], references),
    )
    return create_model(
        "CollectedEvidenceRecommendation",
        __base__=RepairFollowUpRecommendation if sources else AgentRecommendation,
        rationale=(list[rationale], Field(min_length=1 if sources else 0, max_length=12)),
        evidence_ids=(list[evidence_id], Field(max_length=40 if registry else 0)),
        agent_version=(Literal[CLAIMS_AGENT_VERSION], CLAIMS_AGENT_VERSION),
        prompt_version=(Literal[CLAIMS_AGENT_PROMPT_VERSION], CLAIMS_AGENT_PROMPT_VERSION),
    )


def recommendation_constraints(registry):
    sources = repair_discrepancy_sources(registry)
    response_model = _response_model(registry, sources)
    actions = REPAIR_FOLLOW_UP_ACTIONS if sources else tuple(RecommendedNextAction)
    return response_model, {
        "allowed_next_actions": [action.value for action in actions],
        "required_follow_up_evidence_ids": sources,
    }
