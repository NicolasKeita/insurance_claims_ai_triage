"""Labels."""

from __future__ import annotations

from collections import Counter

from vision.dataset import DatasetValidationError
from vision.dataset.cardd_hf.annotations._fields import _get, _kind
from vision.dataset.cardd_hf.annotations._instances import _instances


def _collect_field(raw_samples, name):
    collected = []
    errors = []
    count = 0
    source_count = mask_count = polyline_count = annotated_samples = 0
    source_classes: Counter[str] = Counter()
    for sample in raw_samples:
        value = _get(sample, name)
        if value is None:
            collected.append(())
            continue
        kind = _kind(value)
        source_labels = (
            _get(value, "detections" if kind == "Detections" else "polylines", [])
            if kind in ("Detections", "Polylines")
            else [value]
            if kind in ("Detection", "Polyline")
            else []
        )
        source_count += len(source_labels)
        annotated_samples += bool(source_labels)
        mask_count += sum(
            _kind(label) == "Detection" and (_get(label, "mask") is not None or bool(_get(label, "mask_path")))
            for label in source_labels
        )
        polyline_count += sum(_kind(label) == "Polyline" for label in source_labels)
        source_classes.update(
            _get(label, "label").strip() for label in source_labels if isinstance(_get(label, "label"), str)
        )
        try:
            instances = _instances(value, name)
            collected.append(instances)
            count += len(instances)
        except (DatasetValidationError, OSError, ValueError, TypeError) as exc:
            collected.append(())
            if len(errors) < 3:
                errors.append(str(exc))
    return collected, count, errors, source_count, source_classes, annotated_samples, mask_count, polyline_count
