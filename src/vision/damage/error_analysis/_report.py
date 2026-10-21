"""Report."""

from __future__ import annotations

from typing import Any

from vision.damage.error_analysis._models import ErrorExamplesConfig
from vision.damage.error_analysis._predictions import _inspect_example
from vision.damage.error_analysis._selection import _example_kinds, _ExampleSelection
from vision.damage.training import DamageTrainingConfig, write_json


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
    names = {item["yolo_class_id"]: item["name"] for item in manifest["category_mapping"]}
    images = sorted(manifest["prepared_files"]["test"], key=lambda entry: entry["image"])[: examples.max_images]
    kinds = _example_kinds(examples)
    totals = dict.fromkeys(kinds, 0)
    selected = []
    selection = _ExampleSelection(directory, names, kinds, examples)
    for entry in images:
        image_path, detected, truth, events = _inspect_example(model, config, entry, confidence_threshold, examples)
        selection.consider(entry, image_path, detected, truth, events)
    totals, selected = selection.totals, selection.selected
    metadata = _example_metadata(images, examples, confidence_threshold, totals, selected)
    write_json(directory / "metadata.json", metadata)
    return {key: value for key, value in metadata.items() if key != "examples"} | {"examples_saved": len(selected)}


def _example_metadata(images, examples, confidence_threshold, totals, selected):
    floor = min(0.05, confidence_threshold)
    return {
        "split": "test",
        "images_inspected": len(images),
        "maximum_images": examples.max_images,
        "selection": "first filename-sorted test images; bounded diagnostic sample, not a representative estimate",
        "confidence_threshold": confidence_threshold,
        "diagnostic_candidate_threshold": floor,
        "matching_box_iou": 0.5,
        "diagnostic_polygon_mask_iou": 0.5,
        "incomplete_mask_definition": "Same-class box match with source-resolution polygon intersection/"
        "truth mask area <0.8; fixed diagnostic criterion, not TEST-tuned."
        if examples.include_part_errors
        else None,
        "wrong_part_definition": "Unmatched prediction/GT pair of different classes with box IoU >=0.5; "
        "plausible classification confusion, not causal proof."
        if examples.include_part_errors
        else None,
        "counts_in_inspected_sample": totals,
        "good_prediction_definition": "Same-class one-to-one box IoU >=0.5 and source-resolution rasterized "
        "polygon mask IoU >=0.5; other damage in the same image may still be missed.",
        "limitations": "Greedy same-class box IoU FP/FN approximation, not exact mask FP/FN or AP. "
        "Segmentation errors are box matches with prepared-polygon raster IoU <0.5; "
        "rasterization and bridge/hole boundaries differ from framework masks. "
        "Localization errors use unmatched box IoU [0.1,0.5). "
        "Low-confidence predictions below candidate threshold are unavailable. No threshold tuning on TEST.",
        "examples": selected,
    }
