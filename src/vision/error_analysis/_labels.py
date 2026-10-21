"""Label/prediction instance parsing and error overlay rendering."""

from pathlib import Path
from typing import Any

from vision.error_analysis._models import ErrorInstance


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
