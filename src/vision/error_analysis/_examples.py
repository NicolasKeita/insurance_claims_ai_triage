"""Bounded error-overlay selection and metadata artifacts."""

from dataclasses import asdict
from typing import Any

from vision.error_analysis._labels import _annotate
from vision.error_analysis._matching import _inspect_image
from vision.error_analysis._metadata import build_metadata
from vision.error_analysis._models import ErrorExamplesConfig
from vision.training import DamageTrainingConfig, write_json

BASE_KINDS = (
    "good_predictions",
    "false_positives",
    "false_negatives",
    "low_confidence",
    "localization_errors",
    "segmentation_errors",
)
PART_KINDS = (
    "good_predictions",
    "wrong_parts",
    "incomplete_masks",
    "false_positives",
    "false_negatives",
    "low_confidence",
    "localization_errors",
    "segmentation_errors",
)


def save_error_examples(
    model: Any,
    config: DamageTrainingConfig,
    manifest: dict[str, Any],
    confidence_threshold: float,
    *,
    examples: ErrorExamplesConfig | None = None,
) -> dict[str, Any]:
    """
    One overlay per selected image can show several observed events.

    Other event kinds remain recorded, but each quota selects a distinct image instead of one busy image
    exhausting all quotas.
    """
    examples = examples or ErrorExamplesConfig()
    directory = config.output_dir / "error_examples"
    directory.mkdir(parents=True, exist_ok=False)
    context = _context(config, manifest, examples, confidence_threshold)
    selected, totals = _select_examples(model, context, directory)
    metadata = build_metadata(context, totals, selected)
    write_json(directory / "metadata.json", metadata)
    return {key: value for key, value in metadata.items() if key != "examples"} | {"examples_saved": len(selected)}


def _context(config, manifest, examples, confidence_threshold):
    images = sorted(manifest["prepared_files"]["test"], key=lambda entry: entry["image"])[: examples.max_images]
    return {
        "config": config,
        "root": config.dataset_yaml.parent,
        "names": {item["yolo_class_id"]: item["name"] for item in manifest["category_mapping"]},
        "images": images,
        "kinds": PART_KINDS if examples.include_part_errors else BASE_KINDS,
        "examples": examples,
        "threshold": confidence_threshold,
        "floor": min(0.05, confidence_threshold),
    }


def _select_examples(model, context, directory):
    kinds = context["kinds"]
    max_examples = context["examples"].max_examples
    totals = dict.fromkeys(kinds, 0)
    selected_per_kind = dict.fromkeys(kinds, 0)
    selected = []
    for entry in context["images"]:
        events, detected, truth, image_path = _inspect_image(model, entry, context)
        for kind in kinds:
            totals[kind] += len(events[kind])
        if len(selected) >= max_examples:
            continue
        kind = _next_kind(events, selected_per_kind, kinds, max_examples)
        if kind is None:
            continue
        annotated_name = f"{len(selected) + 1:03d}_{kind}_{image_path.stem}.jpg"
        _annotate(image_path, directory / annotated_name, detected, truth, context["names"])
        selected.append(
            {
                "filename": entry["image"],
                "kind": kind,
                "kinds": [key for key in kinds if events[key]],
                "annotated_filename": annotated_name,
                "predictions": [asdict(value) for value in detected],
                "ground_truth": [asdict(value) for value in truth],
                "events": events,
            }
        )
        selected_per_kind[kind] += 1
    return selected, totals


def _next_kind(events, selected_per_kind, kinds, max_examples):
    quota = max(1, max_examples // len(kinds))
    for kind in kinds:
        if events[kind] and selected_per_kind[kind] < quota:
            return kind
    return None
