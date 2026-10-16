from pathlib import Path

import pytest
from PIL import Image
from test_vision_fusion import damage, part

from claims.case import ClaimCase
from claims.enums import DamageType
from claims.models import Claim
from vision.detector import FakeDamageDetector
from vision.images import ImageValidationError
from vision.models import AssociationStatus, ClaimMultimodalVisionAssessment
from vision.multimodal_service import ClaimMultimodalVisionService
from vision.parts_detector import FakeVehiclePartDetector
from vision.taxonomy import VehiclePartType


def case_with_images(tmp_path, *, filenames=("synthetic.png",), declared=(DamageType.HOOD,)):
    claim = Claim.model_validate_json(Path("data/CLAIM-2026-00001/claim.json").read_text(encoding="utf-8"))
    case = ClaimCase(
        claim=claim.model_copy(update={"images": list(filenames), "declared_damage": list(declared)}), root_dir=tmp_path
    )
    case.images_dir.mkdir()
    for filename in filenames:
        Image.new("RGB", (20, 12)).save(case.image_path(filename))
    return case


def detectors(filenames, *, kind=VehiclePartType.HOOD, has_damage=True):
    return (
        FakeDamageDetector({name: [damage()] if has_damage else [] for name in filenames}),
        FakeVehiclePartDetector({name: [part(kind=kind)] for name in filenames}),
    )


def test_declared_and_quote_matching_visual_damage_are_supported(tmp_path):
    case = case_with_images(tmp_path)
    damage_detector, part_detector = detectors(case.claim.images)
    result = ClaimMultimodalVisionService(damage_detector, part_detector).assess_claim(
        case, quoted_damage_types=[DamageType.HOOD]
    )
    assert result.consistency.visually_supported_declared_damage == [DamageType.HOOD]
    assert result.consistency.visually_supported_quote_damage == [DamageType.HOOD]
    assert result.consistency.additional_visual_damage == []
    assert result.consistency.declared_damage_not_visually_confirmed == []
    assert result.consistency.declared_damage_types == [DamageType.HOOD]
    assert result.consistency.quoted_damage_types == [DamageType.HOOD]


def test_additional_damage_aggregates_types_without_deduplicating_observations_between_photos(tmp_path):
    case = case_with_images(tmp_path, filenames=("first.png", "second.png"), declared=(DamageType.FRONT_BUMPER,))
    damage_detector, part_detector = detectors(case.claim.images)
    result = ClaimMultimodalVisionService(damage_detector, part_detector).assess_claim(
        case, quoted_damage_types=[DamageType.HOOD]
    )
    assert result.consistency.additional_visual_damage == [DamageType.HOOD]
    assert result.consistency.visually_supported_quote_damage == [DamageType.HOOD]
    assert result.consistency.declared_damage_not_visually_confirmed == [DamageType.FRONT_BUMPER]
    assert len(result.images) == len(result.observations) == 2
    assert [path.name for path in damage_detector.calls] == list(case.claim.images)
    assert [path.name for path in part_detector.calls] == list(case.claim.images)
    assert result.aggregate_mapped_damage_types == [DamageType.HOOD]
    assert result.aggregate_vehicle_parts == [VehiclePartType.HOOD]


def test_generic_light_part_is_assigned_visually_but_unmapped_to_business_taxonomy(tmp_path):
    case = case_with_images(tmp_path)
    result = ClaimMultimodalVisionService(*detectors(case.claim.images, kind=VehiclePartType.FRONT_LIGHT)).assess_claim(
        case
    )
    assert result.observations[0].association_status is AssociationStatus.ASSIGNED
    assert result.observations[0].vehicle_part is VehiclePartType.FRONT_LIGHT
    assert result.consistency.visual_damage_unmapped == result.observations
    assert result.consistency.additional_visual_damage == []
    assert result.aggregate_mapped_damage_types == []


def test_non_detection_is_information_and_never_generates_a_negative_risk_signal(tmp_path):
    from investigation.policy import INVESTIGATION_POLICY_V3
    from investigation.signals import signals_from_multimodal

    case = case_with_images(tmp_path)
    result = ClaimMultimodalVisionService(*detectors(case.claim.images, has_damage=False)).assess_claim(case)
    assert result.consistency.declared_damage_not_visually_confirmed == [DamageType.HOOD]
    assert result.observations == []
    assert signals_from_multimodal(result, INVESTIGATION_POLICY_V3) == ()


def test_auditable_snapshot_round_trips_with_geometry_versions_and_policy(tmp_path):
    case = case_with_images(tmp_path)
    result = ClaimMultimodalVisionService(*detectors(case.claim.images)).assess_claim(case)
    snapshot = result.model_dump(mode="json", round_trip=True)
    restored = ClaimMultimodalVisionAssessment.model_validate(snapshot)
    assert restored == result
    evidence = restored.observations[0]
    assert evidence.damage_model_metadata.version == evidence.part_model_metadata.version == "test-only"
    assert evidence.fusion_policy_version == "vision_fusion_v1"
    assert evidence.damage_detection.polygon is not None
    assert evidence.primary_association.candidate_part.polygon is not None
    assert evidence.primary_association.damage_containment_ratio == pytest.approx(1)


def test_escaping_claim_image_path_rejected_before_either_model_is_called(tmp_path):
    case = case_with_images(tmp_path)
    case = ClaimCase(claim=case.claim.model_copy(update={"images": ["../outside.png"]}), root_dir=tmp_path)
    damage_detector, part_detector = detectors(case.claim.images)
    with pytest.raises(ImageValidationError, match="outside"):
        ClaimMultimodalVisionService(damage_detector, part_detector).assess_claim(case)
    assert damage_detector.calls == part_detector.calls == []
