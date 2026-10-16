"""Bounded diagnostics; box matching and polygon mask overlap are not mask AP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vision.training import DamageTrainingConfig, write_json


@dataclass(frozen=True)
class ErrorInstance:
    class_id: int
    box: tuple[float, float, float, float]
    confidence: float | None = None
    polygon: tuple[tuple[float, float], ...] = ()


def box_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    overlap *= max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - overlap
    return overlap / union if union else 0.0


def analyze_instances(
    predictions: list[ErrorInstance],
    truth: list[ErrorInstance],
    *,
    confidence_threshold: float,
    matching_iou: float = 0.5,
    image_size: tuple[int, int] | None = None,
    include_part_errors: bool = False,
) -> dict[str, Any]:
    """Greedily match highest-IoU same-class boxes, once per instance.

    FP/FN are diagnostic approximations at a fixed threshold, not the framework's
    mask AP matching. Optional mask IoU rasterizes the prepared YOLO polygons at
    source image resolution, so it is also separate from framework AP matching.
    """
    accepted = [
        index
        for index, pred in enumerate(predictions)
        if pred.confidence is not None and pred.confidence >= confidence_threshold
    ]
    candidates = [
        (box_iou(predictions[pred].box, target.box), pred, target_index)
        for pred in accepted
        for target_index, target in enumerate(truth)
        if predictions[pred].class_id == target.class_id
    ]
    matched_pred, matched_truth, matches = set(), set(), []
    for overlap, pred, target in sorted(
        candidates,
        key=lambda value: (value[0], predictions[value[1]].confidence, -value[1], -value[2]),
        reverse=True,
    ):
        if overlap >= matching_iou and pred not in matched_pred and target not in matched_truth:
            matched_pred.add(pred)
            matched_truth.add(target)
            matches.append({"prediction_index": pred, "truth_index": target, "box_iou": overlap})
    false_positives = [index for index in accepted if index not in matched_pred]
    good_predictions, segmentation_errors, incomplete_masks = [], [], []
    if image_size is not None:
        for match in matches:
            pred = predictions[match["prediction_index"]]
            target = truth[match["truth_index"]]
            overlap = polygon_mask_overlap(pred.polygon, target.polygon, image_size)
            match["polygon_mask_iou"] = overlap["iou"]
            if include_part_errors:
                match["truth_mask_coverage"] = overlap["truth_coverage"]
                if pred.polygon and target.polygon and overlap["truth_coverage"] < 0.8:
                    incomplete_masks.append(match.copy())
            if match["polygon_mask_iou"] >= matching_iou:
                good_predictions.append(match.copy())
            else:
                segmentation_errors.append(match.copy())
    else:
        good_predictions = [match.copy() for match in matches]
    result = {
        "matches": matches,
        "good_predictions": good_predictions,
        "segmentation_errors": segmentation_errors,
        "false_positives": false_positives,
        "false_negatives": [index for index in range(len(truth)) if index not in matched_truth],
        "low_confidence": [
            index
            for index, pred in enumerate(predictions)
            if pred.confidence is not None and pred.confidence < confidence_threshold
        ],
        "localization_errors": [
            {"prediction_index": pred, "truth_index": target, "box_iou": overlap}
            for overlap, pred, target in candidates
            if pred in false_positives and target not in matched_truth and 0.1 <= overlap < matching_iou
        ],
    }
    if include_part_errors:
        result["wrong_parts"] = [
            {
                "prediction_index": pred,
                "truth_index": target,
                "box_iou": box_iou(predictions[pred].box, truth[target].box),
                "predicted_class_id": predictions[pred].class_id,
                "truth_class_id": truth[target].class_id,
            }
            for pred in false_positives
            for target in result["false_negatives"]
            if predictions[pred].class_id != truth[target].class_id
            and box_iou(predictions[pred].box, truth[target].box) >= matching_iou
        ]
        result["incomplete_masks"] = incomplete_masks
    return result


def polygon_mask_iou(
    first: tuple[tuple[float, float], ...],
    second: tuple[tuple[float, float], ...],
    image_size: tuple[int, int],
) -> float:
    return polygon_mask_overlap(first, second, image_size)["iou"]


def polygon_mask_overlap(
    first: tuple[tuple[float, float], ...],
    second: tuple[tuple[float, float], ...],
    image_size: tuple[int, int],
) -> dict[str, float]:
    """Diagnostic overlap of normalized polygons, rasterized with Pillow.

    This does not recreate Ultralytics' resized prototype masks or AP matcher.
    Missing polygons have zero overlap. Multi-component bridge edges and hole
    boundaries follow the prepared YOLO polygon representation.
    """
    from PIL import Image, ImageChops, ImageDraw

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Diagnostic mask size must be positive")
    masks = [Image.new("1", image_size) for _ in range(2)]
    try:
        for polygon, mask in zip((first, second), masks):
            if len(polygon) >= 3:
                ImageDraw.Draw(mask).polygon([(x * width, y * height) for x, y in polygon], fill=1)
        intersection = ImageChops.logical_and(*masks)
        union = ImageChops.logical_or(*masks)
        try:
            intersection_count = sum(intersection.histogram()[1:])
            union_count = sum(union.histogram()[1:])
            truth_count = sum(masks[1].histogram()[1:])
            return {
                "iou": intersection_count / union_count if union_count else 0.0,
                "truth_coverage": intersection_count / truth_count if truth_count else 0.0,
            }
        finally:
            intersection.close()
            union.close()
    finally:
        for mask in masks:
            mask.close()


def _label_instances(path: Path) -> list[ErrorInstance]:
    instances = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = line.split()
        coordinates = [float(value) for value in values[1:]]
        polygon = tuple(zip(coordinates[::2], coordinates[1::2]))
        xs, ys = coordinates[::2], coordinates[1::2]
        instances.append(ErrorInstance(int(values[0]), (min(xs), min(ys), max(xs), max(ys)), polygon=polygon))
    return instances


def _plain(value: Any) -> list[Any]:
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _prediction_instances(result: Any) -> list[ErrorInstance]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    coordinates, classes, confidences = _plain(boxes.xyxyn), _plain(boxes.cls), _plain(boxes.conf)
    masks = getattr(result, "masks", None)
    polygons = getattr(masks, "xyn", []) if masks is not None else []
    instances = []
    for index, (box, class_id, confidence) in enumerate(zip(coordinates, classes, confidences)):
        polygon = tuple(tuple(point) for point in _plain(polygons[index])) if index < len(polygons) else ()
        instances.append(ErrorInstance(int(class_id), tuple(box), float(confidence), polygon))
    return instances


def _annotate(
    path: Path, output: Path, predictions: list[ErrorInstance], truth: list[ErrorInstance], names: dict[int, str]
) -> None:
    from PIL import Image, ImageDraw

    with Image.open(path) as original:
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    for kind, instances, color in (("GT", truth, "lime"), ("pred", predictions, "orange")):
        for instance in instances:
            box = (
                instance.box[0] * image.width,
                instance.box[1] * image.height,
                instance.box[2] * image.width,
                instance.box[3] * image.height,
            )
            draw.rectangle(box, outline=color, width=2)
            if len(instance.polygon) >= 3:
                points = [(x * image.width, y * image.height) for x, y in instance.polygon]
                draw.line(points + [points[0]], fill=color, width=2)
            label = f"{kind} {names[instance.class_id]}"
            if instance.confidence is not None:
                label += f" {instance.confidence:.2f}"
            draw.text((box[0], max(0, box[1] - 12)), label, fill=color)
    image.save(output, format="JPEG")


def save_error_examples(
    model: Any,
    config: DamageTrainingConfig,
    manifest: dict[str, Any],
    confidence_threshold: float,
    *,
    max_examples: int = 8,
    max_images: int = 100,
    include_part_errors: bool = False,
) -> dict[str, Any]:
    """
    One overlay per selected image can show several observed events.

    Other event kinds remain recorded, but each quota selects a distinct image instead of one busy image
    exhausting all quotas.
    """
    directory = config.output_dir / "error_examples"
    directory.mkdir(parents=True, exist_ok=False)
    root = config.dataset_yaml.parent
    names = {item["yolo_class_id"]: item["name"] for item in manifest["category_mapping"]}
    images = sorted(manifest["prepared_files"]["test"], key=lambda entry: entry["image"])[:max_images]
    kinds = (
        "good_predictions",
        "false_positives",
        "false_negatives",
        "low_confidence",
        "localization_errors",
        "segmentation_errors",
    )
    if include_part_errors:
        kinds = (
            "good_predictions",
            "wrong_parts",
            "incomplete_masks",
            "false_positives",
            "false_negatives",
            "low_confidence",
            "localization_errors",
            "segmentation_errors",
        )
    totals = dict.fromkeys(kinds, 0)
    selected_per_kind = dict.fromkeys(kinds, 0)
    selected = []
    floor = min(0.05, confidence_threshold)
    for entry in images:
        image_path = root / entry["image"]
        predictions = model.predict(
            source=str(image_path),
            conf=floor,
            imgsz=config.imgsz,
            device=config.device,
            verbose=False,
            save=False,
        )
        detected = _prediction_instances(predictions[0])
        truth = _label_instances(root / entry["label"])
        from PIL import Image

        with Image.open(image_path) as original:
            image_size = original.size
        events = analyze_instances(
            detected,
            truth,
            confidence_threshold=confidence_threshold,
            image_size=image_size,
            include_part_errors=include_part_errors,
        )
        for kind in kinds:
            totals[kind] += len(events[kind])
        eligible = [
            kind for kind in kinds if events[kind] and selected_per_kind[kind] < max(1, max_examples // len(kinds))
        ]
        if eligible and len(selected) < max_examples:
            kind = eligible[0]
            annotated_name = f"{len(selected) + 1:03d}_{kind}_{image_path.stem}.jpg"
            _annotate(image_path, directory / annotated_name, detected, truth, names)
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
    metadata = {
        "split": "test",
        "images_inspected": len(images),
        "maximum_images": max_images,
        "selection": "first filename-sorted test images; bounded diagnostic sample, not a representative estimate",
        "confidence_threshold": confidence_threshold,
        "diagnostic_candidate_threshold": floor,
        "matching_box_iou": 0.5,
        "diagnostic_polygon_mask_iou": 0.5,
        "incomplete_mask_definition": "Same-class box match with source-resolution polygon intersection/"
        "truth mask area <0.8; fixed diagnostic criterion, not TEST-tuned."
        if include_part_errors
        else None,
        "wrong_part_definition": "Unmatched prediction/GT pair of different classes with box IoU >=0.5; "
        "plausible classification confusion, not causal proof."
        if include_part_errors
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
    write_json(directory / "metadata.json", metadata)
    return {key: value for key, value in metadata.items() if key != "examples"} | {"examples_saved": len(selected)}
