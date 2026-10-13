"""Render actual domain-adapter inference on deterministic held-out TEST images."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vision.config import VisionSettings
from vision.dataset import load_prepared_dataset
from vision.detector import DamageDetectorError, UltralyticsDamageDetector
from vision.evaluation import selected_checkpoint
from vision.training import DEFAULT_TRAINING_CONFIG, DamageTrainingConfig, load_training_config, write_json
from vision.visualization import save_annotated_image


def select_test_entries(
    manifest: dict[str, Any], *, maximum_images: int,
    diagnostic_examples: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Use first filename-sorted occurrence of each truth class, then diagnostics.

    Membership and selection are deterministic. Class coverage concerns ground
    truth and does not imply that the model detects each class successfully.
    """
    if maximum_images <= 0:
        raise ValueError("maximum_images must be positive")
    entries = sorted(manifest["prepared_files"]["test"], key=lambda row: row["image"])
    by_path = {entry["image"]: entry for entry in entries}
    selected: list[dict[str, Any]] = []
    chosen: set[str] = set()

    def add(entry: dict[str, Any]) -> None:
        if entry["image"] not in chosen and len(selected) < maximum_images:
            chosen.add(entry["image"])
            selected.append(entry)

    for category in sorted(manifest["category_mapping"], key=lambda row: row["yolo_class_id"]):
        class_id = str(category["yolo_class_id"])
        first = next((entry for entry in entries if int(entry.get("class_counts", {}).get(class_id, 0)) > 0), None)
        if first is not None:
            add(first)
    for example in diagnostic_examples or []:
        if example["filename"] not in by_path:
            raise ValueError("Diagnostic example is absent from the frozen TEST inventory")
        add(by_path[example["filename"]])
    for entry in entries:
        add(entry)
    return selected


def run_test_smoke(config: DamageTrainingConfig, *, maximum_images: int = 12) -> dict[str, Any]:
    manifest = load_prepared_dataset(config.dataset_yaml)
    checkpoint, training = selected_checkpoint(config, manifest)
    metrics_path = config.output_dir / "test_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError("Run the final held-out evaluation before the TEST inference smoke")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (metrics.get("split") != "test"
            or metrics.get("checkpoint_selection_split") != "val"
            or metrics.get("checkpoint_sha256") != training["best_checkpoint_sha256"]
            or metrics.get("dataset") != training["dataset"]):
        raise ValueError("TEST metrics do not identify this run's exact VAL-selected best checkpoint and dataset")
    threshold = VisionSettings.from_env().confidence_threshold
    if threshold != training.get("confidence_threshold") or threshold != metrics.get("confidence_threshold"):
        raise ValueError("Smoke inference threshold must equal the previously configured training/evaluation threshold")
    diagnostics_path = config.output_dir / "error_examples/metadata.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.is_file() else {}
    selected = select_test_entries(
        manifest, maximum_images=maximum_images, diagnostic_examples=diagnostics.get("examples", []),
    )
    if not selected:
        raise ValueError("Prepared dataset has no held-out TEST images")
    destination = config.output_dir / "test_smoke"
    if destination.exists():
        raise FileExistsError(f"Smoke artifacts already exist: {destination}")
    detector = UltralyticsDamageDetector(VisionSettings(
        model_path=checkpoint, confidence_threshold=threshold, device=config.device,
    ))
    destination.mkdir(parents=True, exist_ok=False)
    root = config.dataset_yaml.parent
    names = {str(item["yolo_class_id"]): item["name"] for item in manifest["category_mapping"]}
    predicted_classes: Counter[str] = Counter()
    truth_classes: Counter[str] = Counter()
    observations = []
    for index, entry in enumerate(selected, 1):
        source = root / entry["image"]
        assessment = detector.assess_image(source)
        stem = f"{index:03d}_{source.stem}"
        annotated = destination / f"{stem}_annotated.jpg"
        save_annotated_image(source, assessment, annotated)
        assessment_path = destination / f"{stem}_assessment.json"
        write_json(assessment_path, assessment.model_dump(mode="json"))
        predicted_classes.update(detection.damage_type.value for detection in assessment.detections)
        actual_classes = {names[key]: int(count) for key, count in entry.get("class_counts", {}).items()}
        truth_classes.update(actual_classes)
        observations.append({
            "image": entry["image"], "source_sample_id": entry.get("source_sample_id"),
            "width": assessment.width, "height": assessment.height,
            "ground_truth_class_counts": actual_classes,
            "detection_count": len(assessment.detections),
            "usable_polygon_count": sum(detection.polygon is not None for detection in assessment.detections),
            "missing_polygon_count": sum(detection.polygon is None for detection in assessment.detections),
            "warnings": assessment.warnings,
            "annotated_image": annotated.name, "assessment_json": assessment_path.name,
        })
        print(f"TEST smoke {index}/{len(selected)}: {source.name}; {len(assessment.detections)} detections")
    payload = {
        "status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "test", "checkpoint": "train/weights/best.pt",
        "checkpoint_sha256": training["best_checkpoint_sha256"],
        "checkpoint_selection_split": "val", "confidence_threshold": threshold,
        "threshold_selection": "same configured baseline as training/evaluation; no TEST tuning",
        "device": config.device, "adapter": "UltralyticsDamageDetector",
        "dataset_split_version": manifest.get("split_version"),
        "metrics_scope": metrics.get("metrics_scope"),
        "selection": "First filename-sorted TEST occurrence of each ground-truth class, then existing diagnostic examples, then filename-sorted TEST images; deduplicated and bounded.",
        "images_processed": len(observations),
        "detection_count": sum(item["detection_count"] for item in observations),
        "usable_polygon_count": sum(item["usable_polygon_count"] for item in observations),
        "missing_polygon_count": sum(item["missing_polygon_count"] for item in observations),
        "warning_count": sum(len(item["warnings"]) for item in observations),
        "predicted_class_counts": dict(sorted(predicted_classes.items())),
        "ground_truth_class_counts": dict(sorted(truth_classes.items())),
        "visual_review_status": "PENDING — inspect saved overlays before claiming visual verification",
        "limitations": "Qualitative sanity sample, not an accuracy estimate. Ground-truth class coverage does not guarantee successful predictions of each class. Each counted detection is an observation, not a unique physical damage estimate.",
        "images": observations,
    }
    write_json(destination / "smoke_summary.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--device", help="Override config device, e.g. cpu or 0")
    parser.add_argument("--max-images", type=int, default=12)
    args = parser.parse_args()
    try:
        config = load_training_config(args.config)
        if args.device is not None:
            config = replace(config, device=args.device)
        result = run_test_smoke(config, maximum_images=args.max_images)
    except (OSError, ValueError, RuntimeError, DamageDetectorError) as error:
        parser.exit(2, f"TEST inference smoke not executed/completed: {error}\n")
    print(json.dumps({key: result[key] for key in ("status", "images_processed", "detection_count", "usable_polygon_count", "missing_polygon_count", "warning_count")}, indent=2))
    print(f"Smoke artifacts: {config.output_dir / 'test_smoke'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
