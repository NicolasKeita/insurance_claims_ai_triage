"""Follow-up requirements derived from collected structured assessments."""

from typing import Literal, get_args

from agent.models import (
    AgentToolName,
    EvidenceItem,
    EvidenceSourceType,
    ObservationStatus,
    RecommendedNextAction,
    ToolObservation,
)
from claims.enums import DamageType, TriageWorkflow
from claims.investigation.models import SignalCode

FollowUpAction = Literal[
    RecommendedNextAction.REQUEST_ADDITIONAL_INFORMATION,
    RecommendedNextAction.REFER_TO_EXPERT,
    RecommendedNextAction.REFER_FOR_INVESTIGATION_REVIEW,
]
FOLLOW_UP_ACTIONS = get_args(FollowUpAction)
MissingDataAction = Literal[
    RecommendedNextAction.NO_RECOMMENDATION,
    RecommendedNextAction.REQUEST_ADDITIONAL_INFORMATION,
]
MISSING_DATA_ACTIONS = get_args(MissingDataAction)
ESSENTIAL_FACT_TOOLS = {AgentToolName.GET_CLAIM, AgentToolName.GET_HISTORY}
UNAVAILABLE_STATUSES = {ObservationStatus.NOT_AVAILABLE, ObservationStatus.NOT_FOUND}
FOLLOW_UP_SIGNAL_CODES = {
    SignalCode.ADDITIONAL_QUOTE_DAMAGE,
    SignalCode.UNMAPPED_QUOTE_ITEM,
    SignalCode.RECENT_CLAIM_FREQUENCY,
    SignalCode.ANOMALOUS_PATTERN,
    SignalCode.EXTREME_ANOMALY_PERCENTILE,
    SignalCode.VISUAL_ADDITIONAL_DAMAGE,
}
FOLLOW_UP_TRIAGE_WORKFLOWS = {TriageWorkflow.EXPERT_REVIEW, TriageWorkflow.INVESTIGATION}
FOLLOW_UP_RULES = (
    "Collected structured assessments require clarification or further human review when "
    "a TRIAGE_ASSESSMENT recommends "
    + " or ".join(sorted(FOLLOW_UP_TRIAGE_WORKFLOWS))
    + ", an ANOMALY_ASSESSMENT records is_anomalous=true, or an INVESTIGATION_ASSESSMENT records "
    + ", ".join(sorted(FOLLOW_UP_SIGNAL_CODES))
    + ", or a VISUAL_ASSESSMENT records additional_visual_damage in its consistency assessment. "
    "These follow-up triggers do not establish fraud and apply even below the elevated-priority threshold."
)


def _assessment_result(item):
    data = item.structured_data or {}
    result = data.get("result", data) or {}
    return result if isinstance(result, dict) else {}


def _visual_damage_types(result, field):
    consistency = result.get("consistency")
    if not isinstance(consistency, dict) or not isinstance(consistency.get(field), list):
        return ()
    values = consistency[field]
    return tuple(damage for damage in DamageType if damage in values)


def _requires_follow_up(item):
    result = _assessment_result(item)
    if item.source_type == EvidenceSourceType.TRIAGE_ASSESSMENT:
        return result.get("recommended_workflow") in FOLLOW_UP_TRIAGE_WORKFLOWS
    if item.source_type == EvidenceSourceType.ANOMALY_ASSESSMENT:
        assessment = result.get("assessment", result) or {}
        return isinstance(assessment, dict) and assessment.get("is_anomalous") is True
    if item.source_type == EvidenceSourceType.INVESTIGATION_ASSESSMENT:
        return any(signal.get("code") in FOLLOW_UP_SIGNAL_CODES for signal in result.get("signals", []))
    if item.source_type == EvidenceSourceType.VISUAL_ASSESSMENT:
        return bool(_visual_damage_types(result, "additional_visual_damage"))
    return False


def follow_up_sources(registry):
    """Use typed assessment facts, never instructions in retrieved text or objectives.

    Triage workflow guidance and observed atypicality also require follow-up when
    collected. Counts alone do not create a signal here; the investigation policy
    owns its thresholds. No coverage or fraud decision is made.
    """
    sources = []
    for value in registry.values():
        item = EvidenceItem.model_validate(value)
        if _requires_follow_up(item):
            sources.append(item.evidence_id)
    return sources


def unavailable_essential_tools(observations):
    """Use actual failed factual reads, independently of objectives or benchmark inputs.

    Claim facts and history supply factual context. Advisory model assessments,
    optional perception and empty retrieval results do not replace those facts
    and do not by themselves require abstention. An unattempted read is not an outage.
    """
    tools = []
    for value in observations:
        item = ToolObservation.model_validate(value)
        if item.tool_name in ESSENTIAL_FACT_TOOLS and item.status in UNAVAILABLE_STATUSES:
            if item.tool_name.value not in tools:
                tools.append(item.tool_name.value)
    return tools


def visual_uncertainties(registry):
    """Preserve typed non-detections independently of the generated answer and context budget."""
    unconfirmed = set()
    for value in registry.values():
        item = EvidenceItem.model_validate(value)
        if item.source_type == EvidenceSourceType.VISUAL_ASSESSMENT:
            unconfirmed.update(_visual_damage_types(_assessment_result(item), "declared_damage_not_visually_confirmed"))
    if not unconfirmed:
        return []
    parts = ", ".join(damage.value for damage in DamageType if damage in unconfirmed)
    return [f"Declared damage was not visually confirmed ({parts}); non-detection is not evidence of absence."]
