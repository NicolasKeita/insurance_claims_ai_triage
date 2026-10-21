"""Test contracts."""

import pytest
from pydantic import ValidationError

from agent.models import (
    AgentRecommendation,
)
from tests.agent.graph._support import service
from tests.support.agent_fakes import (
    make_recommendation,
    make_step,
)


@pytest.mark.parametrize("iterations", [0, 9, 10000, True])
def test_unsafe_iteration_configuration_rejected(iterations):
    with pytest.raises(ValueError):
        service([], max_iterations=iterations)


@pytest.mark.parametrize("arguments", [{"claim_id": "CLAIM-SECRET-123"}, {"sql": "SELECT *"}])
def test_planner_arguments_cannot_change_claim_or_add_capabilities(arguments):
    with pytest.raises(ValidationError):
        make_step("GET_CLAIM", arguments)


@pytest.mark.parametrize(
    "action", ["APPROVE_CLAIM", "DENY_CLAIM", "FRAUD_CONFIRMED", "PAYMENT_AUTHORIZED", "CUSTOMER_HIGH_RISK"]
)
def test_final_action_enum_excludes_business_decisions(action):
    output = make_recommendation().model_dump()
    output["recommended_next_action"] = action
    with pytest.raises(ValidationError):
        AgentRecommendation.model_validate(output)


def test_human_review_cannot_be_false():
    with pytest.raises(ValidationError):
        AgentRecommendation.model_validate({**make_recommendation().model_dump(), "human_review_required": False})


def test_claim_search_limits_and_action_specific_arguments():
    with pytest.raises(ValidationError):
        make_step("SEARCH_POLICY", {"query": "q", "limit": 6})
    with pytest.raises(ValidationError):
        make_step("GET_HISTORY", {"query": "q"})
    with pytest.raises(ValidationError):
        make_step("SEARCH_POLICY")
