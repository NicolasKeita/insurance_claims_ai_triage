"""Held-out evaluation and honest parsing of framework metrics."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from vision.training import (
    DamageTrainingConfig, dataset_provenance, file_sha256, verify_class_names, write_json,
)


METRIC_FIELDS = ("precision", "recall", "map50", "map50_95")


def _metric_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if not 0 <= number <= 1:
        raise ValueError(f"Metric outside [0,1]: {number}")
    return number


def _overall_metrics(group: Any) -> dict[str, float | None]:
    attrs = ("mp", "mr", "map50", "map")
    return {key: _metric_number(getattr(group, attr, None)) for key, attr in zip(METRIC_FIELDS, attrs)}


def _class_metrics(group: Any) -> dict[int, dict[str, float | None]]:
    if group is None:
        return {}
    indices = getattr(group, "ap_class_index", [])
    if indices is None:
        return {}
    class_result = getattr(group, "class_result", None)
    if not callable(class_result):
        return {}
    values = {}
    for position, class_id in enumerate(indices):
        row = class_result(position)
        values[int(class_id)] = {
            key: _metric_number(row[index]) if len(row) > index else None
            for index, key in enumerate(METRIC_FIELDS)
        }
    return values


def extract_segmentation_metrics(
    result: Any,
    category_mapping: list[dict[str, Any]],
    split_counts: dict[str, Any],
) -> dict[str, Any]:
    """Use actual box/seg Metric values; absent classes stay null.

    Do not use Metric.maps: Ultralytics fills unobserved classes with global mAP,
    which would misrepresent their available per-class evaluation evidence.
    """
    box, mask = getattr(result, "box", None), getattr(result, "seg", None)
    box_classes, mask_classes = _class_metrics(box), _class_metrics(mask)
    absent = dict.fromkeys(METRIC_FIELDS)
    rows = []
    for category in category_mapping:
        class_id, name = int(category["yolo_class_id"]), category["name"]
        source_annotations = split_counts.get("annotations_per_category", {}).get(name, 0)
        source_images = split_counts.get("images_per_category", {}).get(name, 0)
        converted_annotations = split_counts.get("converted_annotations_per_category", {}).get(
            name, source_annotations - split_counts.get("skipped_annotations_per_category", {}).get(name, 0),
        )
        converted_images = split_counts.get("converted_images_per_category", {}).get(name, source_images)
        rows.append({
            **category,
            "annotation_count": converted_annotations,
            "image_count": converted_images,
            "source_annotation_count": source_annotations,
            "source_image_count": source_images,
            "box": box_classes.get(class_id, absent.copy()),
            "mask": mask_classes.get(class_id, absent.copy()),
            "box_metrics_available": class_id in box_classes,
            "mask_metrics_available": class_id in mask_classes,
        })
    ranked = [row for row in rows if row["mask"]["map50_95"] is not None]
    ranked.sort(key=lambda row: (row["mask"]["map50_95"], row["name"]))
    return {
        "box": _overall_metrics(box), "mask": _overall_metrics(mask),
        "per_class": rows,
        "weakest_classes_by_mask_map50_95": [
            {"name": row["name"], "mask_map50_95": row["mask"]["map50_95"],
             "annotation_count": row["annotation_count"]} for row in ranked[:3]
        ],
        "unavailable_classes": [row["name"] for row in rows if not row["mask_metrics_available"]],
        "metric_semantics": "Ultralytics aggregate precision/recall use its confidence curves; conf filters predictions. Fixed-threshold box counts are separate error-analysis approximations.",
    }


def selected_checkpoint(config: DamageTrainingConfig, manifest: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    summary_path = config.output_dir / "training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"No completed training summary: {summary_path}. Train the baseline first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("checkpoint_selection_split") != "val":
        raise ValueError("Evaluation requires a completed run whose checkpoint was selected on VAL")
    if summary.get("task") != "segment" or summary.get("config", {}).get("version") != config.version:
        raise ValueError("Training task/version does not match this evaluation configuration")
    if summary.get("best_checkpoint") != "train/weights/best.pt":
        raise ValueError("Only the training-selected train/weights/best.pt may be evaluated")
    checkpoint = config.output_dir / summary["best_checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint absent: {checkpoint}")
    if file_sha256(checkpoint) != summary.get("best_checkpoint_sha256"):
        raise ValueError("Selected checkpoint SHA-256 differs from the completed training run")
    if dataset_provenance(config.dataset_yaml, manifest) != summary.get("dataset"):
        raise ValueError("Prepared dataset provenance differs from the training run; evaluation refused")
    return checkpoint, summary


def _write_per_class_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["yolo_class_id", "category_id", "name", "image_count", "annotation_count", "source_annotation_count", "source_image_count"]
    columns += [f"{group}_{metric}" for group in ("box", "mask") for metric in METRIC_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flattened = {key: row[key] for key in columns[:7]}
            flattened.update({f"{group}_{key}": value for group in ("box", "mask") for key, value in row[group].items()})
            writer.writerow(flattened)


def run_evaluation(
    config: DamageTrainingConfig,
    *,
    model_factory: Callable[[str], Any],
    environment_provider: Callable[[], dict[str, Any]],
    dataset_loader: Callable[[Path], dict[str, Any]] | None = None,
    max_error_examples: int = 8,
) -> dict[str, Any]:
    if max_error_examples < 0:
        raise ValueError("max_error_examples must be non-negative")
    if dataset_loader is None:
        from vision.dataset import load_prepared_dataset
        dataset_loader = load_prepared_dataset
    manifest = dataset_loader(config.dataset_yaml)
    checkpoint, summary = selected_checkpoint(config, manifest)
    from vision.config import VisionSettings
    threshold = VisionSettings.from_env().confidence_threshold
    if threshold != summary.get("confidence_threshold"):
        raise ValueError("Confidence threshold differs from training metadata. Configure it before training or calibrate on VAL in a new version; never select it on TEST.")
    evaluation_dir = config.output_dir / "evaluation"
    metrics_path = config.output_dir / "test_metrics.json"
    if evaluation_dir.exists() or metrics_path.exists():
        raise FileExistsError("Held-out evaluation artifacts already exist; choose a new experiment version")
    environment = environment_provider()
    model = model_factory(str(checkpoint))
    if getattr(model, "task", None) != "segment":
        raise ValueError("Selected checkpoint must have task=segment")
    verify_class_names(getattr(model, "names", None), manifest["category_mapping"])
    result = model.val(
        data=str(config.dataset_yaml), split="test", conf=threshold,
        imgsz=config.imgsz, batch=config.batch, device=config.device,
        workers=config.workers, project=str(config.output_dir), name="evaluation",
        exist_ok=False, plots=False,
    )
    # Validator save_dir is observed rather than assumed, to catch auto-increment.
    actual_dir = getattr(result, "save_dir", None)
    if actual_dir is not None and Path(actual_dir).resolve() != evaluation_dir.resolve():
        raise RuntimeError("Evaluation wrote to an unexpected directory")
    environment["device_used"] = str(getattr(model, "device", None)) if getattr(model, "device", None) is not None else None
    payload = {
        "schema_version": 1, "split": "test",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": "train/weights/best.pt",
        "checkpoint_sha256": summary["best_checkpoint_sha256"],
        "checkpoint_selection_split": "val", "confidence_threshold": threshold,
        "threshold_selection": "configured baseline; no TEST threshold tuning",
        "environment": environment, "evaluation_config": config.payload(),
        "dataset": summary["dataset"],
        **extract_segmentation_metrics(result, manifest["category_mapping"], manifest["splits"]["test"]),
    }
    write_json(metrics_path, payload)
    csv_path = config.output_dir / "per_class_metrics.csv"
    _write_per_class_csv(csv_path, payload["per_class"])
    if max_error_examples:
        from vision.error_analysis import save_error_examples
        analysis = save_error_examples(model, config, manifest, threshold, max_examples=max_error_examples)
        payload["error_analysis"] = analysis
        write_json(metrics_path, payload)
    from vision.tracking import vision_tracking_run
    with vision_tracking_run(config, run_id=summary.get("mlflow_run_id")) as tracking:
        if tracking is not None:
            tracking.log_evaluation(payload, [metrics_path, csv_path])
    return payload
