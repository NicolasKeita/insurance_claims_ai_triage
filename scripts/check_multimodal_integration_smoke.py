"""Run both local models on the same prior CarDD demo photos, without evaluation.

Each real photo receives its own synthetic diagnostic ClaimCase. Its declaration,
quote and vehicle/incident metadata are illustrative inputs, not real claim facts.
No training, accuracy evaluation, threshold selection or database writes occur.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys

from claims.case import ClaimCase
from claims.enums import ClaimStatus, ClaimType, CollisionType, DamageType
from claims.models import Claim
from vision.config import ROOT, VisionSettings
from vision.fusion_config import DEFAULT_FUSION_POLICY
from vision.images import SUPPORTED_EXTENSIONS
from vision.models import ClaimMultimodalVisionAssessment

DEFAULT_RUN = ROOT / "artifacts/vision/damage_segmentation_v1"
DEFAULT_OUTPUT = ROOT / "artifacts/vision/step30_final_audit/real_integration_smoke"


@dataclass(frozen=True)
class DiagnosticSource:
    path: Path
    filename: str
    source_sample_id: str | None = None


def select_sources(
    smoke_summary_path: Path, training_summary_path: Path, *, max_images: int = 12,
    images: list[Path] | None = None,
) -> tuple[Path, list[DiagnosticSource]]:
    """Preserve the prior demo order and reject paths escaping its dataset images."""
    import yaml

    if max_images <= 0:
        raise ValueError("max_images must be positive")
    training = json.loads(Path(training_summary_path).read_text(encoding="utf-8"))
    if training.get("status") != "completed":
        raise ValueError("Damage source training summary must record completed training")
    dataset_yaml = Path(training["config"]["dataset_yaml"]).expanduser()
    if not dataset_yaml.is_absolute():
        dataset_yaml = ROOT / dataset_yaml
    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(dataset.get("path", dataset_yaml.parent)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = dataset_yaml.parent / dataset_root
    dataset_root = dataset_root.resolve()
    image_root = (dataset_root / "images").resolve()
    if not image_root.is_dir():
        raise ValueError(f"Dataset images directory unavailable: {image_root}")
    prior = json.loads(Path(smoke_summary_path).read_text(encoding="utf-8"))
    entries = ([{"image": str(path)} for path in images] if images else prior["images"])
    selected, seen = [], set()
    for entry in entries:
        path = Path(entry["image"]).expanduser()
        path = (path if path.is_absolute() else dataset_root / path).resolve()
        if not path.is_relative_to(image_root):
            raise ValueError(f"Diagnostic image is outside the source dataset images directory: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported diagnostic image extension: {path.suffix}")
        if path not in seen:
            selected.append(DiagnosticSource(path, path.relative_to(image_root).as_posix(), entry.get("source_sample_id")))
            seen.add(path)
        if len(selected) == max_images:
            break
    if not selected:
        raise ValueError("No prior diagnostic images were selected")
    return dataset_root, selected


def diagnostic_case(dataset_root: Path, source: DiagnosticSource, index: int, declared: list[DamageType]) -> ClaimCase:
    claim = Claim(
        claim_id=f"DIAGNOSTIC-CARDD-{index:03d}", claim_type=ClaimType.AUTO_COLLISION,
        status=ClaimStatus.NEW,
        incident={"date": date(2000, 1, 1), "location": "SYNTHETIC DIAGNOSTIC CONTEXT",
                  "collision_type": CollisionType.PARKING_DAMAGE, "injuries_declared": False},
        vehicle={"make": "UNKNOWN DIAGNOSTIC", "model": "UNKNOWN DIAGNOSTIC", "year": 2000},
        customer={"customer_id": "SYNTHETIC-DIAGNOSTIC-NO-CUSTOMER"},
        policy={"policy_id": "SYNTHETIC-DIAGNOSTIC-NO-POLICY", "product": "DIAGNOSTIC"},
        declared_damage=declared, documents=[], images=[source.filename],
        repair_estimate={"amount": "0", "currency": "EUR"},
    )
    return ClaimCase(claim=claim, root_dir=dataset_root)


class CountingFactory:
    """Wrap the actual local YOLO factory, keeping one cached detector per model."""

    def __init__(self, kind):
        self.kind, self.active_source = kind, None
        self.load_attempts, self.load_count = 0, 0
        self.calls = []

    def __call__(self, checkpoint):
        from ultralytics import YOLO

        self.load_attempts += 1
        model = YOLO(checkpoint)
        self.load_count += 1
        owner = self

        class CountedModel:
            def __getattr__(self, name):
                return getattr(model, name)

            def predict(self, **options):
                from vision.damage.detector import _list

                record = {"source_image": owner.active_source, "requested_device": options.get("device"),
                          "confidence_threshold": options.get("conf"), "status": "STARTED"}
                owner.calls.append(record)
                try:
                    results = model.predict(**options)
                    observed_device = getattr(getattr(model, "predictor", None), "device", None)
                    record["observed_device"] = str(observed_device) if observed_device is not None else None
                    record["results"] = []
                    for result in results:
                        boxes, masks = result.boxes, getattr(result, "masks", None)
                        classes = _list(boxes.cls) if boxes is not None else []
                        confidences = _list(boxes.conf) if boxes is not None else []
                        bounds = _list(boxes.xyxyn) if boxes is not None else []
                        record["results"].append({
                            "dimensions_hw": list(result.orig_shape),
                            "raw_detection_count": len(classes),
                            "detections": [
                                {"class_id": int(class_id), "class_name": result.names[int(class_id)],
                                 "confidence": confidence, "bounding_box_xyxyn": box}
                                for class_id, confidence, box in zip(classes, confidences, bounds)
                            ],
                            "mask_count": len(masks.data) if masks is not None else 0,
                            "mask_tensor_shape": list(masks.data.shape) if masks is not None else None,
                        })
                    record["status"] = "COMPLETED"
                    return results
                except Exception as error:
                    record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
                    raise

        return CountedModel()

    def summary(self):
        return {"load_attempts": self.load_attempts, "load_count": self.load_count,
                "predict_count": len(self.calls), "calls": self.calls}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-summary", type=Path, default=DEFAULT_RUN / "test_smoke/smoke_summary.json")
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_RUN / "training_summary.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Parent directory; creates a new timestamped run")
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--images", type=Path, nargs="+", help="Paths inside the same dataset images directory, relative to dataset root or absolute")
    parser.add_argument("--damage-model", type=Path)
    parser.add_argument("--part-model", type=Path)
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu")
    parser.add_argument("--declared", nargs="*", choices=[item.value for item in DamageType], default=[])
    parser.add_argument("--quoted", nargs="*", choices=[item.value for item in DamageType], default=[])
    args = parser.parse_args(argv)
    run = args.output_dir / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "STARTED", "scope": "DIAGNOSTIC INTEGRATION SMOKE; not a real claim assessment or accuracy evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(), "output_dir": str(run.resolve()),
        "python_executable": sys.executable, "python_version": sys.version,
        "source_smoke_summary": str(args.smoke_summary.resolve()), "source_training_summary": str(args.training_summary.resolve()),
        "context": {"real_declaration_known": False, "real_quote_known": False,
                    "synthetic_declared_damage_types": args.declared, "synthetic_quoted_damage_types": args.quoted,
                    "meaning": "All consistency and investigation signals are conditional on explicit synthetic inputs; they are not facts about real claims."},
        "training_executed": False, "accuracy_evaluation_executed": False, "test_tuning_performed": False,
        "independent_joint_damage_part_ground_truth": False, "fusion_precision_metrics": None,
        "fusion_policy": DEFAULT_FUSION_POLICY.model_dump(mode="json"), "images": [],
    }
    try:
        dataset_root, sources = select_sources(args.smoke_summary, args.training_summary, max_images=args.max_images, images=args.images)
        from claims.investigation.policy import INVESTIGATION_POLICY_V3
        from claims.investigation.signals import signals_from_multimodal
        from vision.damage.detector import UltralyticsDamageDetector
        from vision.multimodal_service import ClaimMultimodalVisionService
        from vision.parts.detector import UltralyticsVehiclePartDetector, VehiclePartSettings
        from vision.visualization import save_multimodal_annotated_image

        damage_env, part_env = VisionSettings.from_env(), VehiclePartSettings.from_env()
        damage_settings = VisionSettings(model_path=args.damage_model or damage_env.model_path,
                                         confidence_threshold=damage_env.confidence_threshold, device=args.device)
        part_settings = VehiclePartSettings(model_path=args.part_model or part_env.model_path,
                                            confidence_threshold=part_env.confidence_threshold, device=args.device)
        damage_factory, part_factory = CountingFactory("damage"), CountingFactory("part")
        service = ClaimMultimodalVisionService(
            UltralyticsDamageDetector(damage_settings, model_factory=damage_factory),
            UltralyticsVehiclePartDetector(part_settings, model_factory=part_factory), DEFAULT_FUSION_POLICY,
        )
        summary.update(dataset_root=str(dataset_root), requested_device=args.device,
                       image_selection="Prior demo order, deduplicated and bounded; labels are not loaded for this smoke",
                       source_split_note="Prior CarDD demo files may belong to its historical TEST split; this run performs diagnostic inference only, without evaluating or tuning any model.",
                       declared_context_is_synthetic=True, quote_context_is_synthetic=True)
        statuses, methods = Counter(), Counter()
        for index, source in enumerate(sources, 1):
            record = {"source_image": str(source.path), "filename": source.filename,
                      "source_sample_id": source.source_sample_id, "status": "STARTED", "stage": "inference"}
            summary["images"].append(record)
            damage_factory.active_source = part_factory.active_source = str(source.path)
            case = diagnostic_case(dataset_root, source, index, [DamageType(item) for item in args.declared])
            try:
                assessment = service.assess_claim(case, quoted_damage_types=[DamageType(item) for item in args.quoted])
                record["stage"] = "snapshot_validation"
                snapshot = assessment.model_dump(mode="json", round_trip=True)
                ClaimMultimodalVisionAssessment.model_validate(snapshot)
                name = f"{index:03d}_{source.path.stem}"
                write_json(run / f"{name}_assessment.json", snapshot)
                record.update(assessment=str(run / f"{name}_assessment.json"), stage="synthetic_context_signals")
                signals = signals_from_multimodal(assessment, INVESTIGATION_POLICY_V3)
                write_json(run / f"{name}_synthetic_context_signals.json", [item.model_dump(mode="json") for item in signals])
                record.update(synthetic_context_signals=str(run / f"{name}_synthetic_context_signals.json"), stage="visualization")
                panel = save_multimodal_annotated_image(source.path, assessment.images[0], run / f"{name}_fused.jpg")
                observations = assessment.observations
                statuses.update(item.association_status.value for item in observations)
                methods.update(candidate.association_method.value for item in observations for candidate in item.associations)
                record.update(status="COMPLETED", stage="completed", diagnostic_claim_id=case.claim.claim_id,
                              damage_detections=assessment.images[0].damage_assessment.detection_count,
                              part_detections=assessment.images[0].part_assessment.detection_count,
                              observations=len(observations), association_statuses=dict(Counter(item.association_status.value for item in observations)),
                              assessment=str(run / f"{name}_assessment.json"), fused_panel=str(panel),
                              synthetic_context_signals=str(run / f"{name}_synthetic_context_signals.json"))
            except Exception as error:
                record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            write_json(run / "summary.json", summary)
            print(f"{index}/{len(sources)} {source.filename}: {record['status']}", flush=True)
        summary.update(damage_model=service.damage_detector.model_metadata.model_dump(mode="json"),
                       part_model=service.part_detector.model_metadata.model_dump(mode="json"),
                       damage_inference_threshold=damage_settings.confidence_threshold, part_inference_threshold=part_settings.confidence_threshold,
                       damage_factory=damage_factory.summary(), part_factory=part_factory.summary(),
                       association_statuses=dict(statuses), association_methods=dict(methods), images_selected=len(sources),
                       images_completed=sum(item["status"] == "COMPLETED" for item in summary["images"]))
        complete = summary["images_completed"] == len(sources)
        expected_sources = Counter(str(source.path) for source in sources)
        once = all(factory.load_count == 1 and Counter(call["source_image"] for call in factory.calls) == expected_sources
                   for factory in (damage_factory, part_factory))
        summary["observed_devices"] = sorted({call["observed_device"] for factory in (damage_factory, part_factory)
                                               for call in factory.calls if call.get("observed_device") is not None})
        summary.update(single_cached_model_and_one_inference_per_image=once, status="COMPLETED" if complete and once else "FAILED")
    except Exception as error:
        summary.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    write_json(run / "summary.json", summary)
    print(f"{summary['status']}: {run / 'summary.json'}", flush=True)
    return 0 if summary["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
