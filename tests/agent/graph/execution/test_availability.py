"""Test availability."""

from agent.models import (
    ObservationStatus,
    RecommendedNextAction,
    ToolObservation,
)
from tests.agent.graph._constants import CLAIM_ID
from tests.agent.graph._support import service
from tests.infrastructure.support.agent_fakes import FakeAgentTools


def test_unavailable_tool_is_preserved_as_uncertainty():
    tools = FakeAgentTools(
        observations={
            "GET_ANOMALY": ToolObservation(
                tool_name="GET_ANOMALY", status="NOT_AVAILABLE", message="No stored anomaly assessment"
            )
        }
    )
    agent, _, _ = service(["GET_CLAIM", "GET_ANOMALY", "FINISH"], tools=tools)
    result = agent.review_claim(CLAIM_ID)
    assert result.trace[1].observation_status == ObservationStatus.NOT_AVAILABLE
    assert "GET_ANOMALY: No stored anomaly assessment" in result.recommendation.uncertainties


def test_no_evidence_abstains_explicitly_without_made_up_finalizer_answer():
    tools = FakeAgentTools(
        observations={
            "GET_CLAIM": ToolObservation(
                tool_name="GET_CLAIM", status="NOT_AVAILABLE", message="Claim data temporarily unavailable"
            )
        }
    )
    agent, _, finalizer = service(["GET_CLAIM", "FINISH"], tools=tools)
    result = agent.review_claim(CLAIM_ID)
    assert result.recommendation.recommended_next_action == RecommendedNextAction.NO_RECOMMENDATION
    assert result.recommendation.evidence_ids == []
    assert not finalizer.calls
