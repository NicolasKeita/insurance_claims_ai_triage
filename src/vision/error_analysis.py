"""Small diagnostic sample, using box overlap rather than exact mask matching."""

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
    predictions: list[ErrorInstance], truth: list[ErrorInstance], *,
    confidence_threshold: float, matching_iou: float = 0.5,
) -> dict[str, Any]:
    """Greedily match highest-IoU same-class boxes, once per instance.

    FP/FN are diagnostic approximations at a fixed threshold, not the framework's
    mask AP matching. Localization includes unmatched same-class overlap <0.5.
    """
    accepted = [index for index, pred in enumerate(predictions) if pred.confidence is not None and pred.confidence >= confidence_threshold]
    candidates = [
        (box_iou(predictions[pred].box, target.box), pred, target_index)
        for pred in accepted for target_index, target in enumerate(truth)
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
    return {
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": [index for index in range(len(truth)) if index not in matched_truth],
        "low_confidence": [index for index, pred in enumerate(predictions) if pred.confidence is not None and pred.confidence < confidence_threshold],
        "localization_errors": [
            {"prediction_index": pred, "truth_index": target, "box_iou": overlap}
            for overlap, pred, target in candidates
            if pred in false_positives and target not in matched_truth and 0.1 <= overlap < matching_iou
        ],
    }


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


def _annotate(path: Path, output: Path, predictions: list[ErrorInstance], truth: list[ErrorInstance], names: dict[int, str]) -> None:
    from PIL import Image, ImageDraw
    with Image.open(path) as original:
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    for kind, instances, color in (("GT", truth, "lime"), ("pred", predictions, "orange")):
        for instance in instances:
            box = (instance.box[0] * image.width, instance.box[1] * image.height,
                   instance.box[2] * image.width, instance.box[3] * image.height)
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
    model: Any, config: DamageTrainingConfig, manifest: dict[str, Any],
    confidence_threshold: float, *, max_examples: int = 8, max_images: int = 100,
) -> dict[str, Any]:
    directory = config.output_dir / "error_examples"
    directory.mkdir(parents=True, exist_ok=False)
    root = config.dataset_yaml.parent
    names = {item["yolo_class_id"]: item["name"] for item in manifest["category_mapping"]}
    images = sorted(manifest["prepared_files"]["test"], key=lambda entry: entry["image"])[:max_images]
    kinds = ("false_positives", "false_negatives", "low_confidence", "localization_errors")
    totals = dict.fromkeys(kinds, 0)
    selected_per_kind = dict.fromkeys(kinds, 0)
    selected = []
    floor = min(0.05, confidence_threshold)
    for entry in images:
        image_path = root / entry["image"]
        predictions = model.predict(
            source=str(image_path), conf=floor, imgsz=config.imgsz,
            device=config.device, verbose=False, save=False,
        )
        detected = _prediction_instances(predictions[0])
        truth = _label_instances(root / entry["label"])
        events = analyze_instances(detected, truth, confidence_threshold=confidence_threshold)
        for kind in kinds:
            totals[kind] += len(events[kind])
            if events[kind] and selected_per_kind[kind] < max(1, max_examples // len(kinds)) and len(selected) < max_examples:
                annotated_name = f"{len(selected) + 1:03d}_{kind}_{image_path.stem}.jpg"
                _annotate(image_path, directory / annotated_name, detected, truth, names)
                selected.append({
                    "filename": entry["image"], "kind": kind, "annotated_filename": annotated_name,
                    "predictions": [asdict(value) for value in detected],
                    "ground_truth": [asdict(value) for value in truth], "events": events,
                })
                selected_per_kind[kind] += 1
    metadata = {
        "split": "test", "images_inspected": len(images), "maximum_images": max_images,
        "selection": "first filename-sorted test images; bounded diagnostic sample, not a representative estimate",
        "confidence_threshold": confidence_threshold, "diagnostic_candidate_threshold": floor,
        "matching_box_iou": 0.5, "counts_in_inspected_sample": totals,
        "limitations": "Greedy same-class box IoU approximation; no exact mask FP/FN. Localization errors use unmatched box IoU [0.1,0.5). Low-confidence predictions below candidate threshold are unavailable. No threshold tuning on TEST.",
        "examples": selected,
    }
    write_json(directory / "metadata.json", metadata)
    return {key: value for key, value in metadata.items() if key != "examples"} | {"examples_saved": len(selected)}
