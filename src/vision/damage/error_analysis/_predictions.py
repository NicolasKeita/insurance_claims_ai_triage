"""Predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision.damage.error_analysis._matching import analyze_instances
from vision.damage.error_analysis._models import AnalysisConfig, ErrorInstance


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


def _inspect_example(model, config, entry, confidence_threshold, examples):
    root = config.dataset_yaml.parent
    floor = min(0.05, confidence_threshold)
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
        config=AnalysisConfig(
            confidence_threshold=confidence_threshold,
            image_size=image_size,
            include_part_errors=examples.include_part_errors,
        ),
    )
    return image_path, detected, truth, events
