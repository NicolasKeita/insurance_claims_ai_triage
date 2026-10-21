"""Test ranking."""

import copy

import pytest
from pydantic import ValidationError

from vision.models import (
    ImageVisualAssessment,
)


def test_removing_a_competing_part_cannot_turn_an_ambiguous_observation_into_certainty():
    from tests.intelligence.vision.fusion.association._support import damage, fused, part

    payload = fused([damage()], [part(), part()]).model_dump(round_trip=True)
    payload["evidence"][0]["associations"].pop()
    with pytest.raises(ValidationError, match="candidates differ"):
        ImageVisualAssessment.model_validate(payload)


def test_forced_assignment_of_ambiguous_masks_is_rejected():
    from tests.intelligence.vision.fusion.association._support import damage, fused, part

    payload = fused([damage()], [part(), part()]).model_dump(round_trip=True)
    evidence = payload["evidence"][0]
    primary = copy.deepcopy(evidence["associations"][0])
    evidence.update(
        association_status="assigned",
        primary_association=primary,
        vehicle_part="hood",
        mapped_damage_type="HOOD",
        part_confidence=0.9,
        containment_ratio=1,
    )
    with pytest.raises(ValidationError, match="ranking and ambiguity"):
        ImageVisualAssessment.model_validate(payload)


def test_lower_containment_primary_cannot_replace_the_clear_best_part():
    from tests.intelligence.vision.fusion.association._support import damage, fused, part, rectangle
    from vision.taxonomy import VehiclePartType

    payload = fused(
        [damage(rectangle(0.1, 0.1, 0.3, 0.3))],
        [
            part(),
            part(rectangle(0.1, 0.1, 0.26, 0.3), kind=VehiclePartType.FRONT_BUMPER),
        ],
    ).model_dump(round_trip=True)
    evidence = payload["evidence"][0]
    primary = copy.deepcopy(evidence["associations"][1])
    evidence.update(
        primary_association=primary,
        vehicle_part="front_bumper",
        mapped_damage_type="FRONT_BUMPER",
        part_confidence=primary["part_confidence"],
        containment_ratio=primary["damage_containment_ratio"],
    )
    with pytest.raises(ValidationError, match="ranking and ambiguity"):
        ImageVisualAssessment.model_validate(payload)
