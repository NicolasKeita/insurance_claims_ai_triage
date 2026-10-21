"""Models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vision.dataset.cardd_hf._constants import Polygon


@dataclass(frozen=True)
class HfDamageInstance:
    label: str
    polygons: tuple[Polygon, ...]
    annotation_field: str = ""
    source_kind: str = "polyline"
    mask_hole_count: int = 0


@dataclass(frozen=True)
class HfDamageSample:
    sample_id: str
    filepath: Path
    width: int
    height: int
    instances: tuple[HfDamageInstance, ...]
    image_sha256: str


@dataclass(frozen=True)
class HfDamageDataset:
    repository: str
    revision: str | None
    samples: tuple[HfDamageSample, ...]
    field_schema: dict[str, str]
    annotation_fields: tuple[str, ...]
    classes: tuple[str, ...]
    source_format: str = "FiftyOneDataset"
    annotation_field_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class CarddHfConfig:
    """Dataset source, cache and injectable loading dependencies."""

    repository: str | None = None
    revision: str | None = None
    cache_dir: Path | str | None = None
    annotation_field: str | None = None
    loader: Callable[..., Any] | None = None
    revision_resolver: Callable[[str, str | None], str] | None = None
