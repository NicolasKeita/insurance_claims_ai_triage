"""One inference per model/photo, with detector output checked before normalization."""

from collections.abc import Iterable
from pathlib import Path

from claims.assets import ImageAsset, resolve_image_assets
from claims.case import ClaimCase
from claims.enums import DamageType
from vision.damage.detector import DamageDetector
from vision.fusion import fuse_image_assessments
from vision.fusion_config import DEFAULT_FUSION_POLICY, VisionFusionPolicy
from vision.images import ImageValidationError
from vision.models import (
    ClaimMultimodalVisionAssessment,
    ImageDamageAssessment,
    ImagePartAssessment,
    ImageVisualAssessment,
)
from vision.multimodal_service._consistency import assess_multimodal_consistency
from vision.parts.detector import VehiclePartDetector


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
        """Fuse every claim image asset with its independent damage and part observations."""
        image_root = case.images_dir.resolve()
        images = [self._assess_asset(asset, image_root) for asset in resolve_image_assets(case)]
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

    def _assess_asset(self, asset: ImageAsset, image_root: Path) -> ImageVisualAssessment:
        if not asset.path.resolve().is_relative_to(image_root):
            raise ImageValidationError(f"Claim image is outside the claim image directory: {asset.filename}")
        damage = self.damage_detector.assess_image(asset.path)
        parts = self.part_detector.assess_image(asset.path)
        self._validate_source(damage, parts, asset)
        source = {"filename": asset.filename, "asset_reference": str(asset.path)}
        return fuse_image_assessments(
            damage.model_copy(update=source),
            parts.model_copy(update=source),
            damage_model_metadata=self.damage_detector.model_metadata,
            policy=self.fusion_policy,
        )

    @staticmethod
    def _validate_source(
        damage: ImageDamageAssessment,
        parts: ImagePartAssessment,
        asset: ImageAsset,
    ) -> None:
        if damage.filename != asset.path.name or parts.filename != asset.path.name:
            raise ImageValidationError("Detector assessment filename does not match the requested image")
        for assessment in (damage, parts):
            if (
                assessment.asset_reference is not None
                and Path(assessment.asset_reference).resolve() != asset.path.resolve()
            ):
                raise ImageValidationError("Detector assessment asset reference does not match the requested image")
