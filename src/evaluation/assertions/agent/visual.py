"""Keep available visual evidence and observed non-detections in real agent reviews."""

from pydantic import ValidationError

from agent.models import EvidenceItem, EvidenceSourceType
from agent.review_policy import visual_uncertainties
from evaluation.assertions.recording import Check


def _collected_visual_evidence(agent, claim_id):
    evidence = []
    for value in agent.get("evidence", []):
        try:
            item = EvidenceItem.model_validate(value)
        except (ValidationError, ValueError):
            continue
        if item.source_type != EvidenceSourceType.VISUAL_ASSESSMENT:
            continue
        data = item.structured_data or {}
        result = data.get("result", data)
        if not isinstance(result, dict):
            continue
        claims = [value for value in (data.get("claim_id"), result.get("claim_id")) if value is not None]
        if isinstance(claim_id, str) and claim_id and claims and all(value == claim_id for value in claims):
            evidence.append(item.evidence_id)
    return evidence


def _check_non_detection_uncertainty(recorder, vision):
    observed = EvidenceItem(
        evidence_id="vision:observed",
        source_type=EvidenceSourceType.VISUAL_ASSESSMENT,
        source_reference="observed",
        summary="Observed visual assessment for the reviewed claim.",
        structured_data=vision,
    )
    required = visual_uncertainties({observed.evidence_id: observed})
    if required:
        recommendation = recorder.observation.agent.get("recommendation") or {}
        actual = recommendation.get("uncertainties", [])
        recorder.record(
            Check(
                "agent:visual_non_detection_uncertainty",
                "agent",
                required,
                actual,
                isinstance(actual, list) and all(value in actual for value in required),
            )
        )


def check_agent_visual(recorder):
    """Require evidence from the reviewed claim only after successful real image assessment."""
    observation = recorder.observation
    vision = observation.vision
    component = observation.components.get("vision")
    if (
        recorder.mode != "real"
        or not observation.agent
        or component is None
        or component.status != "PASS"
        or not isinstance(vision, dict)
        or not vision.get("images")
    ):
        return
    claim_id = vision.get("claim_id")
    collected = _collected_visual_evidence(observation.agent, claim_id)
    recorder.record(
        Check(
            "agent:visual_assessment_collected",
            "agent",
            {"claim_id": claim_id, "source_type": EvidenceSourceType.VISUAL_ASSESSMENT.value},
            {"evidence_ids": collected},
            bool(collected),
        )
    )
    _check_non_detection_uncertainty(recorder, vision)
