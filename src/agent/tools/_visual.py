from agent import tools as adapters
from agent.models import (
    EvidenceItem,
    EvidenceSourceType,
)

from ._data import MAX_SIGNALS, _bounded


def _read_visual(self, claim_id, tool):
    with self.factory() as session:
        record = adapters.VisionAssessmentRepository(session).latest_for_claim(claim_id)
    if record is None:
        return self._missing(tool, "No stored visual assessment is available", unavailable=True)
    visual = record.result
    data = _visual_data(self, record, visual)
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


def _visual_data(self, record, visual):
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
        "consistency": _bounded(consistency),
        "damage_model_metadata": _bounded(visual.damage_model_metadata.model_dump(mode="json")),
        "part_model_metadata": _bounded(visual.part_model_metadata.model_dump(mode="json")),
        "fusion_policy": visual.fusion_policy.model_dump(mode="json"),
        "observations": _bounded(observations),
        "observation_count": len(visual.observations),
        "omitted_observation_count": max(0, len(visual.observations) - MAX_SIGNALS),
        "unmapped_observation_count": len(visual.consistency.visual_damage_unmapped),
    }
