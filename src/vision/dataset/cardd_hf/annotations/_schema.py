"""Schema."""

from __future__ import annotations

from vision.dataset import DatasetValidationError
from vision.dataset.cardd_hf._models import HfDamageInstance
from vision.dataset.cardd_hf.annotations._fields import _schema_kind
from vision.dataset.cardd_hf.annotations._labels import _collect_field


def _annotation_schema(dataset, annotation_field):
    try:
        schema = dataset.get_field_schema()
    except (AttributeError, TypeError) as exc:
        raise DatasetValidationError("Expected a loaded FiftyOne dataset with a field schema") from exc
    field_schema = {name: str(value) for name, value in schema.items()}
    kinds = {name: _schema_kind(value) for name, value in schema.items()}
    candidates = [
        name
        for name, kind in kinds.items()
        if kind in {"Detection", "Detections", "Polyline", "Polylines", "Segmentation"}
    ]
    if annotation_field is not None and annotation_field not in candidates:
        raise DatasetValidationError(
            f"Requested annotation field {annotation_field!r} is absent or not a supported label type; "
            f"schema={field_schema}"
        )
    return field_schema, kinds, candidates


def _inspect_fields(raw_samples, candidates, kinds):
    per_field: dict[str, list[tuple[HfDamageInstance, ...]]] = {}
    summaries = {}
    for name in candidates:
        collected, count, errors, source_count, source_classes, annotated_samples, mask_count, polyline_count = (
            _collect_field(raw_samples, name)
        )
        usable = bool(count) and not errors
        summaries[name] = {
            "kind": kinds[name],
            "usable_for_instance_segmentation": usable,
            "source_labeled_instance_count": source_count,
            "source_instances_per_class": dict(sorted(source_classes.items())),
            "samples_with_annotations": annotated_samples,
            "per_instance_mask_count": mask_count,
            "polyline_count": polyline_count,
            "validated_instance_count": count,
            "errors": errors,
        }
        if usable:
            per_field[name] = collected
    return per_field, summaries


def _select_field(annotation_field, per_field, summaries, field_schema):
    if annotation_field is not None:
        if annotation_field not in per_field:
            raise DatasetValidationError(
                f"Selected annotation field {annotation_field!r} has unusable instance segmentation: "
                f"{summaries.get(annotation_field)}"
            )
        selected = annotation_field
    elif len(per_field) == 1:
        selected = next(iter(per_field))
    elif len(per_field) > 1:
        raise DatasetValidationError(
            f"Multiple usable annotation fields discovered: {sorted(per_field)}. "
            "Select --annotation-field explicitly to avoid duplicate annotations."
        )
    else:
        raise DatasetValidationError(
            f"CarDD HF mirror lacks usable per-instance segmentation masks/polylines. No bbox fallback. "
            f"Inspected schema={field_schema}; annotation findings={summaries}"
        )
    return selected
