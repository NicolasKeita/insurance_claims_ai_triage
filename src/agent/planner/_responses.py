"""Select the response schema from application-owned collected evidence."""

from agent.models import AgentRecommendation, RecommendedNextAction
from agent.review_policy import REPAIR_FOLLOW_UP_ACTIONS, RepairFollowUpAction, repair_discrepancy_sources


class RepairFollowUpRecommendation(AgentRecommendation):
    recommended_next_action: RepairFollowUpAction


def recommendation_constraints(registry):
    sources = repair_discrepancy_sources(registry)
    response_model = RepairFollowUpRecommendation if sources else AgentRecommendation
    actions = REPAIR_FOLLOW_UP_ACTIONS if sources else tuple(RecommendedNextAction)
    return response_model, {
        "allowed_next_actions": [action.value for action in actions],
        "required_follow_up_evidence_ids": sources,
    }
