import math

from vision.fusion._overlap import (
    require_valid_part_polygons,
)
from vision.models import (
    DamagePartAssociation,
    ImageVisualAssessment,
)

from ._association import damage_part_candidates, select_primary_association


def _candidates_match(candidate: DamagePartAssociation, observed: DamagePartAssociation) -> None:
    if candidate.association_method is not observed.association_method or candidate.eligible != observed.eligible:
        raise ValueError("Association method or eligibility differs from its masks and fusion policy")
    for metric in ("intersection_area", "damage_area", "part_area", "damage_containment_ratio", "iou"):
        if not math.isclose(getattr(candidate, metric), getattr(observed, metric), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"Association {metric} differs from the preserved source masks")


def validate_image_association_snapshot(image: ImageVisualAssessment) -> None:
    """Check persisted metrics/candidates against the original observed masks."""
    require_valid_part_polygons(image.part_assessment)
    for item in image.evidence:
        expected = damage_part_candidates(
            item.damage_detection,
            item.damage_detection_index,
            image.part_assessment.detections,
            item.fusion_policy,
        )
        expected_by_index = {candidate.part_detection_index: candidate for candidate in expected}
        actual_by_index = {candidate.part_detection_index: candidate for candidate in item.associations}
        if actual_by_index.keys() != expected_by_index.keys():
            raise ValueError("Preserved association candidates differ from positive overlaps of the source masks")
        for index, candidate in actual_by_index.items():
            _candidates_match(candidate, expected_by_index[index])
        status, primary = select_primary_association(expected, item.fusion_policy)
        primary_index = item.primary_association.part_detection_index if item.primary_association is not None else None
        if item.association_status is not status or primary_index != (
            primary.part_detection_index if primary is not None else None
        ):
            raise ValueError("Association status or primary part differs from containment ranking and ambiguity policy")
