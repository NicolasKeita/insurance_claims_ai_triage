"""Check existing local damage/part checkpoints on real claim images, without downloads."""

import argparse
import json
from pathlib import Path

from claims.assets import resolve_document_assets, resolve_image_assets
from claims.consistency import map_quote_damage_types
from claims.document_extraction import DocumentExtractionError, extract_garage_quote
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType
from claims.loader import load_claim_case
from claims.validation import ClaimFileValidationError
from claims.investigation.policy import INVESTIGATION_POLICY_V3
from claims.investigation.signals import signals_from_multimodal
from vision.config import ROOT, VisionSettings
from vision.damage.detector import DamageDetectorError, UltralyticsDamageDetector
from vision.fusion_config import DEFAULT_FUSION_POLICY, VisionFusionPolicy
from vision.images import ImageValidationError, load_validated_image
from vision.multimodal_service import ClaimMultimodalVisionService
from vision.parts.detector import UltralyticsVehiclePartDetector, VehiclePartSettings
from vision.visualization import save_multimodal_annotated_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-dir", type=Path, default=ROOT / "data/CLAIM-2026-00001")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/vision/multimodal_reference_claim")
    parser.add_argument("--damage-model", type=Path)
    parser.add_argument("--part-model", type=Path)
    parser.add_argument("--fusion-config", type=Path, help="Versioned fusion policy TOML")
    parser.add_argument("--device", help="Explicit device for both detectors, e.g. 0 or cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "execution_status.json"

    def unavailable(reasons, *, claim_id=None):
        payload = {"status": "NOT EXECUTED", "claim_id": claim_id, "reasons": reasons}
        status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for reason in reasons:
            print(f"End-to-end fusion: NOT EXECUTED — {reason}")
        return 2

    try:
        case = load_claim_case(args.claim_dir)
        damage_env, parts_env = VisionSettings.from_env(), VehiclePartSettings.from_env()
        damage_settings = VisionSettings(
            model_path=args.damage_model or damage_env.model_path,
            confidence_threshold=damage_env.confidence_threshold,
            device=args.device or damage_env.device,
        )
        parts_settings = VehiclePartSettings(
            model_path=args.part_model or parts_env.model_path,
            confidence_threshold=parts_env.confidence_threshold,
            device=args.device or parts_env.device,
        )
        policy = VisionFusionPolicy.from_toml(args.fusion_config) if args.fusion_config else DEFAULT_FUSION_POLICY
        assets = resolve_image_assets(case)
        reasons = [] if assets else ["claim has no image assets"]
        image_root = case.images_dir.resolve()
        for asset in assets:
            try:
                if not asset.path.resolve().is_relative_to(image_root):
                    raise ImageValidationError(f"Image outside claim directory: {asset.filename}")
                image = load_validated_image(asset.path)
                print(f"Image: {asset.filename} ({image.width} x {image.height}, EXIF-oriented)")
                image.close()
            except ImageValidationError as error:
                reasons.append(f"unusable image asset: {error}")
        for kind, settings in (("damage", damage_settings), ("part", parts_settings)):
            checkpoint = settings.model_path
            if not checkpoint.is_file() or checkpoint.suffix.casefold() != ".pt" or checkpoint.stat().st_size == 0:
                reasons.append(f"{kind} checkpoint unavailable: {checkpoint}")
        if reasons:
            return unavailable(reasons, claim_id=case.claim.claim_id)

        quoted_damage_types = ()
        quote_status = "No available repair quote; quote support not assessed."
        documents = resolve_document_assets(case)
        for asset in documents:
            if asset.type == DocumentType.REPAIR_QUOTE:
                if not asset.path.resolve().is_relative_to(case.documents_dir.resolve()):
                    raise ValueError("Quote asset is outside the claim document directory")
                quote = extract_garage_quote(extract_pdf_text(asset))
                quoted_damage_types, unmapped = map_quote_damage_types(quote)
                quote_status = f"Mapped quote types: {[value.value for value in quoted_damage_types]}; unmapped: {list(unmapped)}"
                break
        # Each adapter loads its checkpoint lazily once, and each photo is inferred once per model.
        service = ClaimMultimodalVisionService(
            UltralyticsDamageDetector(damage_settings), UltralyticsVehiclePartDetector(parts_settings), policy,
        )
        assessment = service.assess_claim(case, quoted_damage_types=quoted_damage_types)
        signals = signals_from_multimodal(assessment, INVESTIGATION_POLICY_V3)
        print(assessment.model_dump_json(indent=2))
        print("Multimodal investigation signals (investigation_policy_v3):")
        for signal in signals:
            print(signal.model_dump_json(indent=2))
        for index, (asset, observed) in enumerate(zip(assets, assessment.images), 1):
            destination = args.output_dir / f"{index:03d}_{Path(asset.filename).stem}_fused.jpg"
            save_multimodal_annotated_image(asset.path, observed, destination)
            print(f"Fused visualization: {destination}")
        assessment_path = args.output_dir / "claim_multimodal_vision_assessment.json"
        assessment_path.write_text(assessment.model_dump_json(indent=2, round_trip=True) + "\n", encoding="utf-8")
        payload = {
            "status": "COMPLETED", "claim_id": case.claim.claim_id,
            "assessment": str(assessment_path), "quote_status": quote_status,
            "investigation_policy_version": INVESTIGATION_POLICY_V3.version,
            "multimodal_signals": [signal.model_dump(mode="json") for signal in signals],
        }
        status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0
    except (ImageValidationError, DamageDetectorError, ClaimFileValidationError,
            DocumentExtractionError, FileNotFoundError, ImportError, ValueError) as error:
        return unavailable([str(error)])


if __name__ == "__main__":
    raise SystemExit(main())
