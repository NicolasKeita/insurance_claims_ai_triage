"""Test grounding."""

import pytest

from agent.planner import StructuredAgentFinalizer, StructuredAgentPlanner
from agent.service import AgentRunError, AgentServiceConfig, ClaimsAgentService
from tests.agent.safety._constants import CLAIM_ID
from tests.agent.safety._support import make_service
from tests.support.agent_fakes import (
    FakeAgentTools,
    FakeStructuredLlm,
    make_recommendation,
    make_step,
)


def test_generated_versions_must_match_current_agent():
    recommendation = make_recommendation()
    recommendation.prompt_version = "unrecognized_prompt_version"
    service = make_service(["GET_CLAIM", "FINISH"], recommendation=recommendation)
    with pytest.raises(AgentRunError) as caught:
        service.review_claim(CLAIM_ID)
    assert caught.value.failure_kind == "GROUNDING"


def test_rationale_citations_are_aggregated_even_when_llm_omits_top_level_ids():
    recommendation = make_recommendation()
    recommendation.evidence_ids = []
    service = make_service(["GET_CLAIM", "FINISH"], recommendation=recommendation)
    result = service.review_claim(CLAIM_ID)
    assert result.recommendation.evidence_ids == [f"claim:{CLAIM_ID}"]


def test_omitted_finalizer_grounding_fields_are_invalid_generation():
    service = ClaimsAgentService(
        AgentServiceConfig(
            FakeAgentTools(),
            StructuredAgentPlanner(FakeStructuredLlm([make_step("GET_CLAIM"), make_step("FINISH")])),
            StructuredAgentFinalizer(
                FakeStructuredLlm(
                    [
                        {
                            "case_summary": "A claim needs review.",
                            "recommended_next_action": "CONTINUE_STANDARD_REVIEW",
                        }
                    ]
                )
            ),
        )
    )
    with pytest.raises(AgentRunError) as caught:
        service.review_claim(CLAIM_ID)
    assert caught.value.failure_kind == "GENERATION"
