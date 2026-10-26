"""Check visual evidence contracts directly, without executing any scenario pipeline."""

import pytest

from evaluation.assertions.agent import check_agent_expectations
from evaluation.assertions.agent.visual import check_agent_visual
from evaluation.assertions.recording import AssertionRecorder
from evaluation.models import ComponentResult, EvaluationScenario, PipelineObservation

CLAIM_ID = "E2E-NONDETECTION-0001"
UNCERTAINTY = "Declared damage was not visually confirmed (LEFT_HEADLIGHT); non-detection is not evidence of absence."


def visual_evidence(*, claim_id=CLAIM_ID, wrapped=False, source_type="VISUAL_ASSESSMENT"):
    data = {"claim_id": claim_id, "consistency": {"declared_damage_not_visually_confirmed": ["LEFT_HEADLIGHT"]}}
    return {
        "evidence_id": "vision:stored",
        "source_type": source_type,
        "source_reference": "stored-assessment",
        "summary": "Stored visual assessment.",
        "structured_data": {"result": data} if wrapped else data,
    }


def recorder(*, evidence=(), uncertainties=(), mode="real", images=True, vision_status="PASS"):
    scenario = EvaluationScenario(scenario_id="visual_agent_review", description="Review.", claim_source="unused")
    observation = PipelineObservation(
        agent={
            "evidence": list(evidence),
            "recommendation": {"uncertainties": list(uncertainties)},
            "tools_used": ["GET_CLAIM", "GET_INVESTIGATION", "GET_TRIAGE"],
        },
        vision={
            "claim_id": CLAIM_ID,
            "images": [{"filename": "intact.png"}] if images else [],
            "consistency": {"declared_damage_not_visually_confirmed": ["LEFT_HEADLIGHT"]},
        },
        components={"vision": ComponentResult(status=vision_status)} if vision_status is not None else {},
    )
    return AssertionRecorder(scenario, observation, mode)


def checks(value):
    check_agent_visual(value)
    return {item.name: item for item in value.checks}


@pytest.mark.parametrize("tool_name_only", [False, True])
def test_missing_visual_read_fails_even_without_scenario_uncertainty_expectations(tool_name_only):
    value = recorder(uncertainties=["The image associated with the claim is missing."])
    if tool_name_only:
        value.observation.agent["tools_used"].append("GET_VISUAL_ASSESSMENT")
    assert value.scenario.expected_uncertainties == []
    check_agent_expectations(value)
    result = {item.name: item for item in value.checks}
    assert result["agent:visual_assessment_collected"].status == "FAIL"
    assert result["agent:visual_non_detection_uncertainty"].status == "FAIL"


@pytest.mark.parametrize("wrapped", [False, True])
def test_same_claim_visual_evidence_and_preserved_uncertainty_pass(wrapped):
    result = checks(recorder(evidence=[visual_evidence(wrapped=wrapped)], uncertainties=[UNCERTAINTY]))

    assert set(result) == {"agent:visual_assessment_collected", "agent:visual_non_detection_uncertainty"}
    assert all(item.status == "PASS" for item in result.values())


@pytest.mark.parametrize(
    "evidence",
    [
        visual_evidence(claim_id="OTHER-CLAIM"),
        visual_evidence(source_type="CLAIM_DATA"),
        {"evidence_id": "vision:invalid", "source_type": "VISUAL_ASSESSMENT"},
    ],
)
def test_other_claim_or_untyped_evidence_cannot_satisfy_visual_read(evidence):
    result = checks(recorder(evidence=[evidence], uncertainties=[UNCERTAINTY]))

    assert result["agent:visual_assessment_collected"].status == "FAIL"


@pytest.mark.parametrize("uncertainties", [[], ["Some details require human review."]])
def test_generic_uncertainty_cannot_replace_observed_visual_non_confirmation(uncertainties):
    result = checks(recorder(evidence=[visual_evidence()], uncertainties=uncertainties))

    assert result["agent:visual_assessment_collected"].status == "PASS"
    assert result["agent:visual_non_detection_uncertainty"].status == "FAIL"


@pytest.mark.parametrize(
    "mode,images,status",
    [
        ("real", True, None),
        ("real", True, "SKIP"),
        ("real", True, "FAIL"),
        ("real", False, "PASS"),
        ("deterministic", True, "PASS"),
    ],
)
def test_inapplicable_visual_assessment_adds_no_agent_requirement(mode, images, status):
    assert checks(recorder(mode=mode, images=images, vision_status=status)) == {}


def test_absent_vision_adds_no_agent_requirement():
    value = recorder()
    value.observation.vision = None
    assert checks(value) == {}


def test_positive_visual_confirmation_requires_collection_without_non_detection_uncertainty():
    value = recorder(evidence=[visual_evidence()])
    value.observation.vision["consistency"] = {
        "visually_supported_declared_damage": ["LEFT_HEADLIGHT"],
        "declared_damage_not_visually_confirmed": [],
    }

    result = checks(value)

    assert list(result) == ["agent:visual_assessment_collected"]
    assert result["agent:visual_assessment_collected"].status == "PASS"
