"""Follow-up requirements derived from collected repair-assessment facts."""

from typing import Literal, get_args

from agent.models import EvidenceItem, EvidenceSourceType, RecommendedNextAction
from claims.investigation.models import SignalCode

RepairFollowUpAction = Literal[
    RecommendedNextAction.REQUEST_ADDITIONAL_INFORMATION,
    RecommendedNextAction.REFER_TO_EXPERT,
    RecommendedNextAction.REFER_FOR_INVESTIGATION_REVIEW,
]
REPAIR_FOLLOW_UP_ACTIONS = get_args(RepairFollowUpAction)
REPAIR_DISCREPANCY_CODES = {SignalCode.ADDITIONAL_QUOTE_DAMAGE, SignalCode.UNMAPPED_QUOTE_ITEM}


def repair_discrepancy_sources(registry):
    """Use structured investigation facts, never instructions in retrieved text or objectives.

    A recorded unexplained quote item warrants clarification or expert review even
    below the elevated-priority threshold. No coverage or fraud decision is made.
    """
    sources = []
    for value in registry.values():
        item = EvidenceItem.model_validate(value)
        if item.source_type != EvidenceSourceType.INVESTIGATION_ASSESSMENT:
            continue
        data = item.structured_data or {}
        result = data.get("result", data) or {}
        signals = result.get("signals", [])
        if any(signal.get("code") in REPAIR_DISCREPANCY_CODES for signal in signals):
            sources.append(item.evidence_id)
    return sources
