"""Follow-up requirements derived from collected investigation facts."""

from typing import Literal, get_args

from agent.models import EvidenceItem, EvidenceSourceType, RecommendedNextAction
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
}


def follow_up_sources(registry):
    """Use structured investigation facts, never instructions in retrieved text or objectives.

    Recorded repair discrepancies or recent claim frequency require follow-up even
    below the elevated-priority threshold. Counts alone do not create a signal here;
    the investigation policy owns its thresholds. No coverage or fraud decision is made.
    """
    sources = []
    for value in registry.values():
        item = EvidenceItem.model_validate(value)
        if item.source_type != EvidenceSourceType.INVESTIGATION_ASSESSMENT:
            continue
        data = item.structured_data or {}
        result = data.get("result", data) or {}
        signals = result.get("signals", [])
        if any(signal.get("code") in FOLLOW_UP_SIGNAL_CODES for signal in signals):
            sources.append(item.evidence_id)
    return sources
