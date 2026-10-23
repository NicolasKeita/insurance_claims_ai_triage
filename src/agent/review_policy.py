"""Follow-up requirements derived from collected structured assessments."""

from typing import Literal, get_args

from agent.models import EvidenceItem, EvidenceSourceType, RecommendedNextAction
from claims.enums import TriageWorkflow
from claims.investigation.models import SignalCode

FollowUpAction = Literal[
    RecommendedNextAction.REQUEST_ADDITIONAL_INFORMATION,
    RecommendedNextAction.REFER_TO_EXPERT,
    RecommendedNextAction.REFER_FOR_INVESTIGATION_REVIEW,
]
FOLLOW_UP_ACTIONS = get_args(FollowUpAction)
FOLLOW_UP_SIGNAL_CODES = {
    SignalCode.ADDITIONAL_QUOTE_DAMAGE,
    SignalCode.UNMAPPED_QUOTE_ITEM,
    SignalCode.RECENT_CLAIM_FREQUENCY,
    SignalCode.ANOMALOUS_PATTERN,
    SignalCode.EXTREME_ANOMALY_PERCENTILE,
}
FOLLOW_UP_TRIAGE_WORKFLOWS = {TriageWorkflow.EXPERT_REVIEW, TriageWorkflow.INVESTIGATION}
FOLLOW_UP_RULES = (
    "Collected structured assessments require clarification or further human review when "
    "a TRIAGE_ASSESSMENT recommends "
    + " or ".join(sorted(FOLLOW_UP_TRIAGE_WORKFLOWS))
    + ", an ANOMALY_ASSESSMENT records is_anomalous=true, or an INVESTIGATION_ASSESSMENT records "
    + ", ".join(sorted(FOLLOW_UP_SIGNAL_CODES))
    + ". These follow-up triggers do not establish fraud and apply even below the elevated-priority threshold."
)


def _requires_follow_up(item):
    data = item.structured_data or {}
    result = data.get("result", data) or {}
    if not isinstance(result, dict):
        return False
    if item.source_type == EvidenceSourceType.TRIAGE_ASSESSMENT:
        return result.get("recommended_workflow") in FOLLOW_UP_TRIAGE_WORKFLOWS
    if item.source_type == EvidenceSourceType.ANOMALY_ASSESSMENT:
        assessment = result.get("assessment", result) or {}
        return isinstance(assessment, dict) and assessment.get("is_anomalous") is True
    if item.source_type == EvidenceSourceType.INVESTIGATION_ASSESSMENT:
        return any(signal.get("code") in FOLLOW_UP_SIGNAL_CODES for signal in result.get("signals", []))
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
