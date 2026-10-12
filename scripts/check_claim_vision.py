"""Assess real local claim images using an existing trained damage checkpoint."""

import argparse
from pathlib import Path

from claims.assets import resolve_image_assets
from claims.loader import load_claim_case
from claims.validation import ClaimFileValidationError
from vision.config import ROOT, VisionSettings
from vision.detector import DamageDetectorError, UltralyticsDamageDetector, require_local_checkpoint
from vision.images import ImageValidationError, load_validated_image
from vision.service import ClaimVisionService
from vision.visualization import save_annotated_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-dir", type=Path, default=ROOT / "data/CLAIM-2026-00001")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/vision/reference_claim")
    parser.add_argument("--model", type=Path, help="Existing trained local .pt checkpoint")
    parser.add_argument("--confidence-threshold", type=float)
    args = parser.parse_args()
    environment = VisionSettings.from_env()
    settings = VisionSettings(
        model_path=args.model or environment.model_path,
        confidence_threshold=environment.confidence_threshold if args.confidence_threshold is None else args.confidence_threshold,
        device=environment.device,
    )
    try:
        case = load_claim_case(args.claim_dir)
        assets = resolve_image_assets(case)
        if not assets:
            print("Reference claim inference: NOT EXECUTED — claim has no image assets.")
            return 2
        invalid = []
        image_root = case.images_dir.resolve()
        for asset in assets:
            try:
                if not asset.path.resolve().is_relative_to(image_root):
                    raise ImageValidationError(f"Image is outside the claim image directory: {asset.filename}")
                image = load_validated_image(asset.path, settings)
                print(f"Image: {asset.filename} ({image.width} x {image.height}, EXIF-oriented)")
                image.close()
            except ImageValidationError as error:
                invalid.append(str(error))
        if invalid:
            print("Reference claim inference: NOT EXECUTED — unusable image assets:")
            for error in invalid:
                print(f"- {error}")
            return 2
        require_local_checkpoint(settings.model_path)
        assessment = ClaimVisionService(UltralyticsDamageDetector(settings)).assess_claim(case)
        print(assessment.model_dump_json(indent=2))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for asset, image_assessment in zip(assets, assessment.images):
            destination = args.output_dir / f"{Path(asset.filename).stem}_annotated.jpg"
            save_annotated_image(asset.path, image_assessment, destination)
            print(f"Annotated image: {destination}")
        output = args.output_dir / "claim_vision_assessment.json"
        output.write_text(assessment.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return 0
    except (ImageValidationError, DamageDetectorError, ClaimFileValidationError, FileNotFoundError, ImportError) as error:
        print(f"Reference claim inference: NOT EXECUTED — {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
