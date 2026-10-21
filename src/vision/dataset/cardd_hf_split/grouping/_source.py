"""Source."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vision.dataset.cardd_hf import HfDamageDataset

from vision.dataset import DatasetValidationError
from vision.dataset.cardd_hf_split._identity import _digest, _Group
from vision.dataset.cardd_hf_split.grouping._duplicates import (
    _duplicate_groups,
    _duplicate_metadata,
)
from vision.dataset.cardd_hf_split.grouping._samples import (
    _sample_annotations,
    _validate_sample,
)


def _source_context(dataset: HfDamageDataset) -> tuple[dict[str, Any], list[_Group]]:
    if not isinstance(dataset.repository, str) or not dataset.repository.strip():
        raise DatasetValidationError("HF mirror repository must be identified")
    if dataset.revision is not None and (not isinstance(dataset.revision, str) or not dataset.revision.strip()):
        raise DatasetValidationError("HF mirror revision must be a nonempty string or unavailable (null)")
    classes = tuple(dataset.classes)
    if (
        not classes
        or len(set(classes)) != len(classes)
        or any(not isinstance(name, str) or not name.strip() for name in classes)
    ):
        raise DatasetValidationError("HF mirror classes must be unique, nonempty labels")
    samples = sorted(dataset.samples, key=lambda sample: sample.sample_id)
    if len(samples) < 3:
        raise DatasetValidationError("At least three HF mirror samples are needed for train/val/test")
    rows, groups, paths, hashes, names = _source_groups(samples, classes)
    category_mapping = [
        {"category_id": index + 1, "name": name, "yolo_class_id": index} for index, name in enumerate(classes)
    ]
    context = {
        "source_name": f"CarDD HF mirror ({dataset.repository})",
        "source_repo": dataset.repository,
        "source_revision": dataset.revision,
        "source_format": getattr(dataset, "source_format", "FiftyOneDataset"),
        "sample_count": len(samples),
        "official_splits_used": False,
        "annotation_fields": list(dataset.annotation_fields),
        "field_schema": dict(dataset.field_schema),
        "category_mapping": category_mapping,
        "samples": rows,
        "annotation_fingerprint": _digest(
            [{key: value for key, value in row.items() if key != "file_name"} for row in rows]
        ),
        "duplicate_grouping": _duplicate_metadata(groups, paths, hashes, names),
    }
    context["source_fingerprint"] = _digest(context)
    return context, groups


def _source_groups(samples, classes):
    rows, seen_ids = [], set()

    paths, hashes, names = defaultdict(list), defaultdict(list), defaultdict(list)
    sample_labels = {}
    for index, sample in enumerate(samples):
        sample_id, path = _validate_sample(sample, seen_ids)
        annotations, labels, annotation_fingerprint = _sample_annotations(sample, classes)
        rows.append(
            {
                "sample_id": sample_id,
                "file_name": path.name,
                "image_sha256": sample.image_sha256,
                "width": sample.width,
                "height": sample.height,
                "annotation_sha256": annotation_fingerprint,
                "annotation_count": len(annotations),
                "annotations_per_category": dict(sorted(Counter(item["label"] for item in annotations).items())),
            }
        )
        sample_labels[sample_id] = labels
        paths[str(path).casefold()].append(index)
        hashes[sample.image_sha256].append(index)
        names[path.name.casefold()].append(index)
    groups = _duplicate_groups(samples, paths, hashes, sample_labels, classes)

    return rows, groups, paths, hashes, names
