"""Claim, visual-assessment and history observations."""

from agent.models import EvidenceItem, EvidenceSourceType
from agent.tools.bounding import MAX_SIGNALS, bounded, claim_data
from persistence.history import HistoricalClaimRepository
from persistence.vision_assessments import VisionAssessmentRepository


class ClaimObservations:
    """Build the claim, visual and history tool observations."""

    def _claim_observation(self, claim_id, tool):
        claim = self._claim(claim_id)
        data = claim_data(claim)
        evidence = EvidenceItem(
            evidence_id=f"claim:{claim_id}",
            source_type=EvidenceSourceType.CLAIM_DATA,
            source_reference=claim_id,
            summary=(
                f"Current claim: {claim.claim_type.value}, incident {claim.incident.date}; "
                f"repair estimate {claim.repair_estimate.amount} {claim.repair_estimate.currency}; "
                f"{len(claim.documents)} document entries and {len(claim.images)} images."
            ),
            structured_data=data,
        )
        return self._observation(tool, data, [evidence])

    def _visual_observation(self, claim_id, tool):
        with self.factory() as session:
            record = VisionAssessmentRepository(session).latest_for_claim(claim_id)
        if record is None:
            return self._missing(tool, "No stored visual assessment is available", unavailable=True)
        visual = record.result
        data = _visual_data(record, visual)
        evidence = EvidenceItem(
            evidence_id=f"vision:{record.id}",
            source_type=EvidenceSourceType.VISUAL_ASSESSMENT,
            source_reference=str(record.id),
            structured_data=data,
            summary=(
                f"Stored multimodal perception: {len(visual.observations)} per-photo observations; "
                f"{len(visual.consistency.additional_visual_damage)} additional mapped damage categories. "
                "Association is geometric. Non-detection is not evidence of absence; "
                "positive corroboration does not increase risk."
            ),
        )
        return self._observation(tool, data, [evidence])

    def _history_observation(self, claim_id, tool):
        with self.factory() as session:
            profile = HistoricalClaimRepository(session).get_profile(claim_id)
        data = profile.model_dump(mode="json")
        evidence = EvidenceItem(
            evidence_id=f"history:{claim_id}",
            source_type=EvidenceSourceType.HISTORY,
            source_reference=claim_id,
            summary=(
                f"Incident-time historical aggregate: {profile.previous_claim_count} earlier claims; "
                f"{profile.claims_last_30_days} in 30 days, {profile.claims_last_365_days} in 365 days. "
                "Prior anomaly flags are statistical model outputs, not established misconduct."
            ),
            structured_data=data,
        )
        return self._observation(tool, data, [evidence])


def _visual_data(record, visual):
    observations = [
        {
            "image": item.filename,
            "visual_damage_type": item.visual_damage_type.value,
            "vehicle_part": item.vehicle_part.value if item.vehicle_part is not None else None,
            "mapped_damage_type": item.mapped_damage_type.value if item.mapped_damage_type is not None else None,
            "damage_confidence": item.damage_confidence,
            "part_confidence": item.part_confidence,
            "containment_ratio": item.containment_ratio,
            "association_status": item.association_status.value,
        }
        for item in visual.observations[:MAX_SIGNALS]
    ]
    consistency = visual.consistency.model_dump(mode="json", exclude={"visual_damage_unmapped"})
    return {
        "assessment_id": str(record.id),
        "created_at": record.created_at.isoformat(),
        "claim_id": visual.claim_id,
        "consistency": bounded(consistency),
        "damage_model_metadata": bounded(visual.damage_model_metadata.model_dump(mode="json")),
        "part_model_metadata": bounded(visual.part_model_metadata.model_dump(mode="json")),
        "fusion_policy": visual.fusion_policy.model_dump(mode="json"),
        "observations": bounded(observations),
        "observation_count": len(visual.observations),
        "omitted_observation_count": max(0, len(visual.observations) - MAX_SIGNALS),
        "unmapped_observation_count": len(visual.consistency.visual_damage_unmapped),
    }
