"""Selection."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from vision.damage.error_analysis._models import ErrorInstance


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


class _ExampleSelection:
    """Bounded diagnostic selection with one shared budget and per-kind quotas."""

    def __init__(self, directory, names, kinds, examples):
        self.directory, self.names, self.kinds, self.examples = directory, names, kinds, examples
        self.totals = dict.fromkeys(kinds, 0)
        self.selected_per_kind = dict.fromkeys(kinds, 0)
        self.selected = []

    def consider(self, entry, image_path, detected, truth, events):
        for kind in self.kinds:
            self.totals[kind] += len(events[kind])
        eligible = [
            kind
            for kind in self.kinds
            if events[kind] and self.selected_per_kind[kind] < max(1, self.examples.max_examples // len(self.kinds))
        ]
        if eligible and len(self.selected) < self.examples.max_examples:
            kind = eligible[0]
            annotated_name = f"{len(self.selected) + 1:03d}_{kind}_{image_path.stem}.jpg"
            _annotate(image_path, self.directory / annotated_name, detected, truth, self.names)
            self.selected.append(
                {
                    "filename": entry["image"],
                    "kind": kind,
                    "kinds": [key for key in self.kinds if events[key]],
                    "annotated_filename": annotated_name,
                    "predictions": [asdict(value) for value in detected],
                    "ground_truth": [asdict(value) for value in truth],
                    "events": events,
                }
            )
            self.selected_per_kind[kind] += 1


def _example_kinds(examples):
    kinds = (
        "good_predictions",
        "false_positives",
        "false_negatives",
        "low_confidence",
        "localization_errors",
        "segmentation_errors",
    )
    if examples.include_part_errors:
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
    return kinds
