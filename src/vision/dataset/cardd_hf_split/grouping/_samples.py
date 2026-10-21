"""Samples."""

from __future__ import annotations

from pathlib import Path

from vision.dataset import DatasetValidationError, annotation_hash
from vision.dataset.cardd_hf_split._identity import _digest


def _sample_annotations(sample, classes):
    annotations = []
    labels = set()
    for instance in sample.instances:
        if instance.label not in classes:
            raise DatasetValidationError(f"HF mirror sample {sample.sample_id}: annotation uses an unknown class")
        labels.add(instance.label)
        annotations.append(
            {
                "label": instance.label,
                "polygons": instance.polygons,
                "annotation_field": getattr(instance, "annotation_field", None),
                "source_kind": getattr(instance, "source_kind", None),
            }
        )
    try:
        annotation_fingerprint = _digest(annotations)
    except (ValueError, TypeError) as exc:
        raise DatasetValidationError(
            f"HF mirror sample {sample.sample_id}: annotations cannot be fingerprinted"
        ) from exc
    return annotations, labels, annotation_fingerprint


def _validate_sample(sample, seen_ids):
    sample_id = sample.sample_id
    if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen_ids:
        raise DatasetValidationError("HF mirror has an empty or duplicate sample ID")
    seen_ids.add(sample_id)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (sample.width, sample.height)
    ):
        raise DatasetValidationError(f"HF mirror sample {sample_id}: invalid dimensions")
    path = Path(sample.filepath).expanduser().resolve()
    if not path.is_file():
        raise DatasetValidationError(f"HF mirror sample {sample_id}: image file is missing")
    if (
        not isinstance(sample.image_sha256, str)
        or len(sample.image_sha256) != 64
        or any(c not in "0123456789abcdef" for c in sample.image_sha256)
    ):
        raise DatasetValidationError(f"HF mirror sample {sample_id}: invalid image SHA256")
    if annotation_hash(path) != sample.image_sha256:
        raise DatasetValidationError(f"HF mirror sample {sample_id}: image content changed after inspection")
    return sample_id, path
