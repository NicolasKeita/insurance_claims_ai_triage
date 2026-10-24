"""Fixed-threshold counts from a separate framework pass with mask/box matching."""

import csv
from typing import Any

import numpy as np


def _counts(true_positives: int, predictions: int, targets: int) -> dict[str, int | float | None]:
    if true_positives > min(predictions, targets):
        raise RuntimeError("Framework returned inconsistent fixed-threshold matching counts")
    return {
        "tp": true_positives,
        "fp": predictions - true_positives,
        "fn": targets - true_positives,
        "precision": true_positives / predictions if predictions else None,
        "recall": true_positives / targets if targets else None,
    }


class OperatingPointCollector:
    """Consume each image once, before Ultralytics clears raw matching statistics."""

    def __init__(self, mapping: list[dict[str, Any]]):
        self.mapping = mapping
        self.class_ids = {int(row["yolo_class_id"]) for row in mapping}
        self.totals = {
            class_id: {key: 0 for key in ("targets", "predictions", "box_tp", "mask_tp")} for class_id in self.class_ids
        }
        self.seen = 0

    def __call__(self, validator: Any) -> None:
        stats = getattr(getattr(validator, "metrics", None), "stats", None)
        required = ("tp", "tp_m", "pred_cls", "target_cls", "conf")
        if not isinstance(stats, dict) or any(key not in stats for key in required):
            raise RuntimeError("Framework matching statistics unavailable for fixed-threshold VAL diagnostics")
        iou = getattr(validator, "iouv", None)
        if iou is None or len(iou) == 0 or not np.isclose(float(iou[0]), 0.5):
            raise RuntimeError("Fixed-threshold diagnostics require matching at IoU 0.50")
        lengths = {len(stats[key]) for key in required}
        if len(lengths) != 1 or len(stats["pred_cls"]) < self.seen:
            raise RuntimeError("Framework VAL statistics changed unexpectedly during diagnostics")
        for index in range(self.seen, len(stats["pred_cls"])):
            self._image(stats, index)
        self.seen = len(stats["pred_cls"])

    def _image(self, stats: dict[str, Any], index: int) -> None:
        predictions = np.asarray(stats["pred_cls"][index])
        targets = np.asarray(stats["target_cls"][index])
        box = np.asarray(stats["tp"][index])
        mask = np.asarray(stats["tp_m"][index])
        if box.ndim != 2 or mask.ndim != 2 or min(box.shape[1], mask.shape[1]) < 1:
            raise RuntimeError("Framework returned invalid mask/box matching arrays")
        if len(box) != len(predictions) or len(mask) != len(predictions):
            raise RuntimeError("Framework prediction and matching array lengths differ")
        observed = set(predictions.tolist()) | set(targets.tolist())
        if not observed <= self.class_ids:
            raise RuntimeError("Framework matching statistics contain an unknown class")
        for class_id, totals in self.totals.items():
            selected = predictions == class_id
            totals["targets"] += int(np.count_nonzero(targets == class_id))
            totals["predictions"] += int(np.count_nonzero(selected))
            totals["box_tp"] += int(np.count_nonzero(box[selected, 0]))
            totals["mask_tp"] += int(np.count_nonzero(mask[selected, 0]))

    def report(self, expected_images: int) -> dict[str, Any]:
        if self.seen != expected_images:
            raise RuntimeError(f"Fixed-threshold VAL coverage differs: {self.seen} images, expected {expected_images}")
        rows = []
        for category in self.mapping:
            totals = self.totals[int(category["yolo_class_id"])]
            rows.append(
                {
                    "yolo_class_id": int(category["yolo_class_id"]),
                    "name": category["name"],
                    "target_count": totals["targets"],
                    "prediction_count": totals["predictions"],
                    **{
                        group: _counts(totals[f"{group}_tp"], totals["predictions"], totals["targets"])
                        for group in ("box", "mask")
                    },
                }
            )
        predictions = sum(row["prediction_count"] for row in rows)
        targets = sum(row["target_count"] for row in rows)
        return {
            "matching_iou": 0.5,
            "images_evaluated": self.seen,
            "aggregation": (
                "micro totals; class-aware one-to-one Ultralytics matching in a separate fixed-confidence VAL pass"
            ),
            "per_class": rows,
            **{
                group: _counts(sum(row[group]["tp"] for row in rows), predictions, targets) for group in ("box", "mask")
            },
        }


def write_operating_csv(path, rows: list[dict[str, Any]]) -> None:
    columns = ["yolo_class_id", "name", "target_count", "prediction_count"]
    columns += [f"{group}_{key}" for group in ("box", "mask") for key in ("tp", "fp", "fn", "precision", "recall")]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            record = {key: row[key] for key in columns[:4]}
            record.update({f"{group}_{key}": value for group in ("box", "mask") for key, value in row[group].items()})
            writer.writerow(record)
