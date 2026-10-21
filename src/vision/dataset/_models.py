"""Models."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetValidationError(ValueError):
    """An external or prepared dataset violates the required data contract."""


@dataclass(frozen=True)
class CocoSplit:
    name: str
    annotation_path: Path
    image_dir: Path
    document: dict[str, Any]


@dataclass(frozen=True)
class CarddDataset:
    root: Path
    splits: dict[str, CocoSplit]
    category_mapping: list[dict[str, Any]]


@dataclass
class _CocoConversion:
    """Shared category mapping and staged outputs of a COCO conversion."""

    manifest: dict
    class_ids: dict
    unsupported: str
    expected: dict = None
    documents: dict = None

    def __post_init__(self):
        self.expected = {}
        self.documents = {}


@dataclass(frozen=True)
class _ConversionCounts:
    """Annotation omissions and multipart conversions in one source split."""

    skipped: Counter
    skipped_by_class: Counter
    stitched_multipart_count: int


@dataclass(frozen=True)
class _PreparedCounts:
    """Observed image and class inventory for a prepared split."""

    image_count: int
    row_count: int
    class_counts: Counter
    class_images: Counter
