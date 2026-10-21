"""Converter."""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from vision.dataset._files import _write_json
from vision.dataset._models import DatasetValidationError
from vision.dataset._polygons import validate_segmentation_label


def _try_ultralytics_conversion(
    stage: Path,
    documents: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, list[tuple[int, list[float]]]]],
    class_count: int,
) -> None:
    """Use official converter only on prefiltered, remapped polygon annotations."""
    from ultralytics.data.converter import convert_coco

    signature = inspect.signature(convert_coco)
    if not {"labels_dir", "save_dir", "use_segments", "cls91to80"}.issubset(signature.parameters):
        raise DatasetValidationError("Installed Ultralytics converter lacks required custom segmentation parameters")
    input_dir = stage / "converter_annotations"
    for split, document in documents.items():
        _write_json(input_dir / f"instances_{split}.json", document)
    conversion_dir = stage / "converter_output"
    convert_coco(labels_dir=str(input_dir), save_dir=str(conversion_dir), use_segments=True, cls91to80=False)
    validated_files = []
    for split, files in expected.items():
        for filename, expected_rows in files.items():
            path = conversion_dir / "labels" / split / PurePosixPath(filename).with_suffix(".txt")
            if not expected_rows and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
            rows = validate_segmentation_label(path, class_count=class_count, expected_count=len(expected_rows))
            if len(rows) != len(expected_rows) or any(
                a[0] != b[0] or len(a[1]) != len(b[1]) or any(abs(x - y) > 1e-5 for x, y in zip(a[1], b[1]))
                for a, b in zip(rows, expected_rows)
            ):
                raise DatasetValidationError(f"Official converter changed polygon geometry/class mapping: {path}")
            validated_files.append((path, stage / "labels" / split / PurePosixPath(filename).with_suffix(".txt")))
    for source, destination in validated_files:
        shutil.copy2(source, destination)


def _coco_backend(stage, documents, expected, converter, unsupported):
    conversion = {
        "backend": "native_polygon",
        "unsupported_policy": unsupported,
        "segmentation_enabled": True,
        "bbox_fallback": False,
        "multipart_policy": "preserve component vertices using retraced nearest-vertex bridges",
        "multipart_limitation": "YOLO uses one polygon per instance; "
        "disconnected topology and holes cannot be represented exactly, "
        "and rasterization may include bridge pixels",
        "use_segments": True,
        "cls91to80": False,
    }
    if converter != "native":
        try:
            _try_ultralytics_conversion(stage, documents, expected, len(next(iter(documents.values()))["categories"]))
            conversion["backend"] = "ultralytics.convert_coco"
        except Exception as exc:
            if converter == "ultralytics":
                raise DatasetValidationError(f"Official COCO segmentation conversion failed validation: {exc}") from exc
            conversion["fallback_reason"] = (
                f"{type(exc).__name__}: official converter unavailable, incompatible, "
                "or failed strict polygon validation"
            )
    return conversion
