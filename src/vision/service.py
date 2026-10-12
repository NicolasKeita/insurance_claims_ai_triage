"""File-based ClaimCase orchestration; no storage paths are invented."""

from claims.assets import resolve_image_assets
from claims.case import ClaimCase
from vision.detector import DamageDetector
from vision.images import ImageValidationError
from vision.models import ClaimVisionAssessment


class ClaimVisionService:
    def __init__(self, detector: DamageDetector):
        self.detector = detector

    def assess_claim(self, case: ClaimCase) -> ClaimVisionAssessment:
        images = []
        image_root = case.images_dir.resolve()
        for asset in resolve_image_assets(case):
            if not asset.path.resolve().is_relative_to(image_root):
                raise ImageValidationError(f"Claim image is outside the claim image directory: {asset.filename}")
            assessment = self.detector.assess_image(asset.path)
            images.append(assessment.model_copy(update={"filename": asset.filename, "asset_reference": str(asset.path)}))
        return ClaimVisionAssessment(
            claim_id=case.claim.claim_id, images=images,
            model_metadata=self.detector.model_metadata,
            confidence_threshold=self.detector.confidence_threshold,
        )
