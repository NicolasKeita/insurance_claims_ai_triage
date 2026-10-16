"""Claim orchestration: one inference per model/photo and pre-mapped quote types."""

from collections.abc import Iterable
from pathlib import Path

from claims.assets import resolve_image_assets
from claims.case import ClaimCase
from claims.enums import DamageType
from vision.detector import DamageDetector
from vision.fusion import fuse_image_assessments
from vision.fusion_config import DEFAULT_FUSION_POLICY, VisionFusionPolicy
from vision.images import ImageValidationError
from vision.models import (
    ClaimMultimodalVisionAssessment,
    ImageVisualAssessment,
    MultimodalConsistencyAssessment,
)
from vision.parts_detector import VehiclePartDetector


def assess_multimodal_consistency(
    images: Iterable[ImageVisualAssessment],
    *,
    declared_damage_types: Iterable[DamageType],
    quoted_damage_types: Iterable[DamageType] = (),
) -> MultimodalConsistencyAssessment:
    """Compare domain parts only; quote text is never reparsed by vision."""
    declared = {DamageType(item) for item in declared_damage_types}
    quoted = {DamageType(item) for item in quoted_damage_types}
    evidence = [item for image in images for item in image.evidence]
    mapped = {item.mapped_damage_type for item in evidence if item.mapped_damage_type is not None}

    def ordered(values):
        return [damage for damage in DamageType if damage in values]

    return MultimodalConsistencyAssessment(
        declared_damage_types=ordered(declared),
        quoted_damage_types=ordered(quoted),
        visually_supported_declared_damage=ordered(mapped & declared),
        additional_visual_damage=ordered(mapped - declared),
        visually_supported_quote_damage=ordered(mapped & quoted),
        visual_damage_unmapped=[item for item in evidence if item.mapped_damage_type is None],
        declared_damage_not_visually_confirmed=ordered(declared - mapped),
    )


class ClaimMultimodalVisionService:
    def __init__(
        self,
        damage_detector: DamageDetector,
        part_detector: VehiclePartDetector,
        fusion_policy: VisionFusionPolicy | None = None,
    ):
        self.damage_detector = damage_detector
        self.part_detector = part_detector
        self.fusion_policy = fusion_policy or DEFAULT_FUSION_POLICY

    def assess_claim(
        self,
        case: ClaimCase,
        *,
        quoted_damage_types: Iterable[DamageType] = (),
    ) -> ClaimMultimodalVisionAssessment:
        images = []
        image_root = case.images_dir.resolve()
        for asset in resolve_image_assets(case):
            if not asset.path.resolve().is_relative_to(image_root):
                raise ImageValidationError(f"Claim image is outside the claim image directory: {asset.filename}")
            damage = self.damage_detector.assess_image(asset.path)
            parts = self.part_detector.assess_image(asset.path)
            # A detector's output is checked before its source identity is
            # normalized to the ClaimCase asset (nested filenames are supported).
            if damage.filename != asset.path.name or parts.filename != asset.path.name:
                raise ImageValidationError("Detector assessment filename does not match the requested image")
            for assessment in (damage, parts):
                if (
                    assessment.asset_reference is not None
                    and Path(assessment.asset_reference).resolve() != asset.path.resolve()
                ):
                    raise ImageValidationError("Detector assessment asset reference does not match the requested image")
            source = {"filename": asset.filename, "asset_reference": str(asset.path)}
            images.append(
                fuse_image_assessments(
                    damage.model_copy(update=source),
                    parts.model_copy(update=source),
                    damage_model_metadata=self.damage_detector.model_metadata,
                    policy=self.fusion_policy,
                )
            )
        consistency = assess_multimodal_consistency(
            images,
            declared_damage_types=case.claim.declared_damage,
            quoted_damage_types=quoted_damage_types,
        )
        return ClaimMultimodalVisionAssessment(
            claim_id=case.claim.claim_id,
            images=images,
            damage_model_metadata=self.damage_detector.model_metadata,
            part_model_metadata=self.part_detector.model_metadata,
            fusion_policy=self.fusion_policy,
            consistency=consistency,
        )
