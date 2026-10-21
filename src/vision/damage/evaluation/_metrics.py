"""Per-class metric rows, weakest-class ranking and the CSV report."""

import csv
from typing import Any

from vision.damage.evaluation._framework_metrics import METRIC_FIELDS, class_metrics, overall_metrics

_CSV_METADATA_COLUMNS = (
    "yolo_class_id",
    "category_id",
    "name",
    "image_count",
    "annotation_count",
    "source_annotation_count",
    "source_image_count",
)


def _category_row(
    category: dict[str, Any],
    split_counts: dict[str, Any],
    box_classes: dict[int, dict[str, float | None]],
    mask_classes: dict[int, dict[str, float | None]],
) -> dict[str, Any]:
    class_id, name = int(category["yolo_class_id"]), category["name"]
    source_annotations = split_counts.get("annotations_per_category", {}).get(name, 0)
    source_images = split_counts.get("images_per_category", {}).get(name, 0)
    converted_annotations = split_counts.get("converted_annotations_per_category", {}).get(
        name,
        source_annotations - split_counts.get("skipped_annotations_per_category", {}).get(name, 0),
    )
    converted_images = split_counts.get("converted_images_per_category", {}).get(name, source_images)
    absent = dict.fromkeys(METRIC_FIELDS)
    return {
        **category,
        "annotation_count": converted_annotations,
        "image_count": converted_images,
        "source_annotation_count": source_annotations,
        "source_image_count": source_images,
        "box": box_classes.get(class_id, absent.copy()),
        "mask": mask_classes.get(class_id, absent.copy()),
        "box_metrics_available": class_id in box_classes,
        "mask_metrics_available": class_id in mask_classes,
    }


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
    box_classes, mask_classes = class_metrics(box), class_metrics(mask)
    rows = [_category_row(category, split_counts, box_classes, mask_classes) for category in category_mapping]
    ranked = [row for row in rows if row["mask"]["map50_95"] is not None]
    ranked.sort(key=lambda row: (row["mask"]["map50_95"], row["name"]))
    return {
        "box": overall_metrics(box),
        "mask": overall_metrics(mask),
        "per_class": rows,
        "weakest_classes_by_mask_map50_95": [
            {"name": row["name"], "mask_map50_95": row["mask"]["map50_95"], "annotation_count": row["annotation_count"]}
            for row in ranked[:3]
        ],
        "unavailable_classes": [row["name"] for row in rows if not row["mask_metrics_available"]],
        "metric_semantics": "Ultralytics aggregate precision/recall use its confidence curves; "
        "conf filters predictions. Fixed-threshold box counts are separate error-analysis approximations.",
    }


def write_per_class_csv(path: Any, rows: list[dict[str, Any]]) -> None:
    columns = list(_CSV_METADATA_COLUMNS)
    columns += [f"{group}_{metric}" for group in ("box", "mask") for metric in METRIC_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flattened = {key: row[key] for key in _CSV_METADATA_COLUMNS}
            flattened.update(
                {f"{group}_{key}": value for group in ("box", "mask") for key, value in row[group].items()}
            )
            writer.writerow(flattened)
