"""Normalization."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from vision.dataset import DatasetValidationError, annotation_hash
from vision.dataset.cardd_hf._models import HfDamageDataset, HfDamageSample
from vision.dataset.cardd_hf.annotations._fields import _get
from vision.dataset.cardd_hf.annotations._schema import (
    _annotation_schema,
    _inspect_fields,
    _select_field,
)


def normalize_fiftyone_dataset(
    dataset: Any, repository: str, revision: str | None, *, annotation_field: str | None = None
) -> HfDamageDataset:
    """
    Discover schema/usable instance fields from actual loaded FiftyOne data.

    Portable identity does not depend on the local MongoDB object ID.
    """
    field_schema, kinds, candidates = _annotation_schema(dataset, annotation_field)
    raw_samples = list(dataset.iter_samples())
    if not raw_samples:
        raise DatasetValidationError("CarDD HF mirror has no samples")
    per_field, summaries = _inspect_fields(raw_samples, candidates, kinds)
    selected = _select_field(annotation_field, per_field, summaries, field_schema)
    samples = _normalize_samples(raw_samples, per_field, selected)
    classes = tuple(sorted({instance.label for sample in samples for instance in sample.instances}))
    return HfDamageDataset(
        repository,
        revision,
        tuple(sorted(samples, key=lambda sample: sample.sample_id)),
        field_schema,
        (selected,),
        classes,
        annotation_field_summaries=summaries,
    )


def _normalize_samples(raw_samples, per_field, selected):
    from PIL import Image

    samples = []
    ids = set()
    for raw, instances in zip(raw_samples, per_field[selected]):
        filepath = Path(_get(raw, "filepath", "")).expanduser().resolve()
        if not filepath.is_file():
            raise DatasetValidationError(f"Downloaded image is missing: {filepath}")
        try:
            with Image.open(filepath) as image:
                width, height = image.size
                image.verify()
        except (OSError, ValueError) as exc:
            raise DatasetValidationError(f"Cannot read image: {filepath}: {exc}") from exc
        metadata = _get(raw, "metadata")
        if (
            width < 1
            or height < 1
            or (
                metadata is not None
                and (_get(metadata, "width", width) != width or _get(metadata, "height", height) != height)
            )
        ):
            raise DatasetValidationError(f"Actual image dimensions disagree with FiftyOne metadata: {filepath}")
        digest = annotation_hash(filepath)
        coco_id = _get(raw, "coco_id")
        if isinstance(coco_id, int) and not isinstance(coco_id, bool) and coco_id >= 0:
            sample_id = f"coco:{coco_id}"
        else:
            sample_id = "image:" + hashlib.sha256((filepath.name + "\0" + digest).encode()).hexdigest()
        if sample_id in ids:
            raise DatasetValidationError(f"Duplicate stable sample ID: {sample_id}")
        ids.add(sample_id)
        samples.append(HfDamageSample(sample_id, filepath, width, height, instances, digest))
    return samples
