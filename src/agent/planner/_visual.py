"""Stored image availability determines relevant visual evidence reads."""

from agent.models import AgentPlanStep, EvidenceItem, EvidenceSourceType


def claim_image_count(registry, claim_id):
    """Read image availability only from typed facts for the current claim."""
    for value in registry.values():
        item = EvidenceItem.model_validate(value)
        if item.source_type != EvidenceSourceType.CLAIM_DATA or item.source_reference != claim_id:
            continue
        data = item.structured_data or {}
        if data.get("claim_id", claim_id) != claim_id:
            continue
        count = data.get("image_count")
        if type(count) is int and count >= 0:
            return count
    return None


def required_visual_step(state, attempted):
    if claim_image_count(state["evidence_registry"], state["claim_id"]) and "GET_VISUAL_ASSESSMENT" not in attempted:
        return AgentPlanStep(
            action="GET_VISUAL_ASSESSMENT",
            arguments={},
            reason="Read the stored visual assessment for images registered on this claim.",
        )
    return None
