"""Registered images require a stored visual read before discretionary planning."""

import pytest

from agent.models import EvidenceSourceType, ToolObservation
from agent.planner import StructuredAgentFinalizer, StructuredAgentPlanner
from agent.service import AgentServiceConfig, ClaimsAgentService
from tests.agent.review._support import prompt_data
from tests.agent.safety._constants import CLAIM_ID
from tests.infrastructure.support.agent_fakes import (
    FakeAgentTools,
    FakeStructuredLlm,
    make_evidence,
    make_recommendation,
    make_step,
)


def claim_observation(count=1, *, source_type=EvidenceSourceType.CLAIM_DATA, reference=CLAIM_ID):
    item = make_evidence(f"claim:{CLAIM_ID}").model_copy(
        update={
            "source_type": source_type,
            "source_reference": reference,
            "structured_data": {"claim_id": CLAIM_ID, "image_count": count},
        }
    )
    return ToolObservation(tool_name="GET_CLAIM", status="OK", evidence=[item])


def visual_observation(status):
    items = []
    if status == "OK":
        items.append(
            make_evidence("vision:stored").model_copy(
                update={
                    "source_type": EvidenceSourceType.VISUAL_ASSESSMENT,
                    "structured_data": {"consistency": {"declared_damage_not_visually_confirmed": ["LEFT_HEADLIGHT"]}},
                }
            )
        )
    return ToolObservation(tool_name="GET_VISUAL_ASSESSMENT", status=status, evidence=items)


def review(claim, *, visual_status="OK", budget=8, require_investigation=False):
    planner_llm = FakeStructuredLlm(
        [make_step("FINISH")] if require_investigation else [make_step("GET_CLAIM"), make_step("FINISH")]
    )
    recommendation = make_recommendation()
    recommendation.uncertainties = []
    finalizer_llm = FakeStructuredLlm([recommendation])
    tools = FakeAgentTools(
        observations={
            "GET_CLAIM": claim,
            "GET_VISUAL_ASSESSMENT": visual_observation(visual_status),
        }
    )
    service = ClaimsAgentService(
        AgentServiceConfig(
            tools,
            StructuredAgentPlanner(planner_llm, require_investigation=require_investigation),
            StructuredAgentFinalizer(finalizer_llm),
            max_iterations=budget,
        )
    )
    result = service.review_claim(CLAIM_ID)
    return result, tools, planner_llm, finalizer_llm


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("status", ["OK", "NOT_AVAILABLE", "NOT_FOUND"])
def test_registered_images_are_read_even_when_the_model_would_finish(required, status):
    result, tools, _, finalizer = review(claim_observation(), visual_status=status, require_investigation=required)
    actions = [call[1] for call in tools.calls]
    expected = ["GET_CLAIM"]
    if required:
        expected.extend(["GET_INVESTIGATION", "GET_TRIAGE"])
    assert actions == [*expected, "GET_VISUAL_ASSESSMENT"]
    assert result.status == "COMPLETED"
    assert prompt_data(finalizer.calls[0])["claim_image_count"] == 1
    if status == "OK":
        assert result.recommendation.uncertainties == [
            "Declared damage was not visually confirmed (LEFT_HEADLIGHT); non-detection is not evidence of absence."
        ]
    else:
        assert not any(item.source_type == "VISUAL_ASSESSMENT" for item in result.evidence)
        assert any(value.startswith("GET_VISUAL_ASSESSMENT:") for value in result.recommendation.uncertainties)


@pytest.mark.parametrize("count", [0, None, True, "1", -1])
def test_absent_or_untyped_image_counts_do_not_trigger_visual_io(count):
    result, tools, _, _ = review(claim_observation(count))
    assert [call[1] for call in tools.calls] == ["GET_CLAIM"]
    assert result.status == "COMPLETED"


@pytest.mark.parametrize(
    "source_type,reference",
    [
        (EvidenceSourceType.HISTORY, CLAIM_ID),
        (EvidenceSourceType.CLAIM_DATA, "OTHER-CLAIM"),
    ],
)
def test_foreign_or_untrusted_image_claims_cannot_force_a_visual_read(source_type, reference):
    claim = claim_observation(source_type=source_type, reference=reference)
    result, tools, _, _ = review(claim)
    assert [call[1] for call in tools.calls] == ["GET_CLAIM"]
    assert result.status == "COMPLETED"


def test_registered_images_do_not_expand_the_explicit_tool_budget():
    result, tools, _, _ = review(claim_observation(), budget=1)
    assert [call[1] for call in tools.calls] == ["GET_CLAIM"]
    assert result.tool_call_count == result.iteration_count == 1
    assert "Maximum tool steps reached." in result.recommendation.uncertainties
