"""Inspect the CarDD HF mirror through FiftyOne, without eager dependencies.

This source is a subset/mirror, never the official CarDD train/val/test splits.
The normalized contract contains real instance geometry only: no bbox fallback.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from vision.dataset import DatasetValidationError, annotation_hash
from vision.taxonomy import map_category_name

DEFAULT_HF_REPOSITORY = "harpreetsahota/CarDD"
_LOAD_LOCK = threading.RLock()
Point = tuple[float, float]
Polygon = tuple[Point, ...]


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


def default_hf_cache_dir() -> Path:
    configured = os.environ.get("CARDD_HF_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    parent = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache")))
    return (parent / "insurance_claims_ai_triage" / "datasets").resolve()


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    try:
        return value[key]
    except (TypeError, KeyError, AttributeError, ValueError):
        return getattr(value, key, default)


def _kind(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("_cls", "")).rsplit(".", 1)[-1]
    return type(value).__name__


def _schema_kind(value: Any) -> str:
    document_type = getattr(value, "document_type", None)
    if document_type is not None:
        return document_type.__name__
    text = str(value)
    match = re.search(r"\b(Detections|Detection|Polylines|Polyline|Segmentation)\b", text)
    return match.group(1) if match else ""


def _label(value: Any) -> str:
    name = _get(value, "label")
    if not isinstance(name, str) or not name.strip():
        raise DatasetValidationError("Instance annotation has no nonempty class label")
    try:
        map_category_name(name)
    except ValueError as exc:
        raise DatasetValidationError(str(exc)) from exc
    return name.strip()


def _area(points: Polygon | list[Point]) -> float:
    return (
        sum(
            x * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * y
            for i, (x, y) in enumerate(points)
        )
        / 2
    )


def _polygon(points: Any) -> Polygon:
    if not isinstance(points, (tuple, list)):
        raise DatasetValidationError("Instance polygon points must be a sequence")
    result: list[Point] = []
    for point in points:
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise DatasetValidationError("Instance polygon vertices must be x/y pairs")
        if any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1
            for v in point
        ):
            raise DatasetValidationError("Instance polygons require finite normalized coordinates in [0, 1]")
        result.append((float(point[0]), float(point[1])))
    if result and result[-1] == result[0]:
        result.pop()
    if len(set(result)) < 3 or abs(_area(result)) < 1e-12:
        raise DatasetValidationError("Degenerate instance polygon")
    return tuple(result)


def _contains(polygon: Polygon | list[Point], point: Point) -> bool:
    x, y = point
    inside = False
    for i, (a, b) in enumerate(polygon):
        c, d = polygon[(i + 1) % len(polygon)]
        if (b > y) != (d > y) and x < (c - a) * (y - b) / (d - b) + a:
            inside = not inside
    return inside


def _bridge(outer: list[Point], ring: list[Point]) -> list[Point]:
    """Retrace a zero-area connector; retain the hole ring's opposite winding."""
    _, left, right = min(
        ((x - a) ** 2 + (y - b) ** 2, i, j) for i, (x, y) in enumerate(outer) for j, (a, b) in enumerate(ring)
    )
    rotated = ring[right:] + ring[:right]
    return outer[: left + 1] + rotated + [rotated[0], outer[left]] + outer[left + 1 :]


def _mask_polygons(mask: Any, bbox: Any) -> tuple[tuple[Polygon, ...], int]:
    """Trace binary pixel-cell edges, preserving components and hole vertices.

    Holes become retraced bridges in their outer component, matching YOLO's
    single-contour contract. A rasterizer may add bridge pixels; this limitation
    is explicit in inspection metadata. No mask is replaced by its rectangle.
    """
    import numpy as np

    array = np.asarray(mask)
    if array.ndim != 2 or not array.size:
        raise DatasetValidationError("Instance mask must be a nonempty 2D binary array")
    if (
        not (np.issubdtype(array.dtype, np.bool_) or np.issubdtype(array.dtype, np.number))
        or not np.isfinite(array).all()
    ):
        raise DatasetValidationError("Instance mask must contain finite binary pixels")
    unique = set(np.unique(array).tolist())
    if not unique.issubset({0, 1, 255}) or (1 in unique and 255 in unique):
        raise DatasetValidationError("Per-instance masks require binary pixels, not semantic class IDs")
    binary = array != 0
    if not binary.any():
        raise DatasetValidationError("Instance mask contains no foreground pixels")
    if (
        not isinstance(bbox, (tuple, list))
        or len(bbox) != 4
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox)
    ):
        raise DatasetValidationError("Instance mask requires a finite normalized x/y/width/height bounding box")
    x, y, width, height = map(float, bbox)
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > 1 + 1e-9 or y + height > 1 + 1e-9:
        raise DatasetValidationError("Instance mask bounding box lies outside normalized image coordinates")
    padded = np.pad(binary, 1)
    boundaries = (
        binary & ~padded[:-2, 1:-1],
        binary & ~padded[1:-1, 2:],
        binary & ~padded[2:, 1:-1],
        binary & ~padded[1:-1, :-2],
    )
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for side, boundary in enumerate(boundaries):
        for row, col in zip(*np.nonzero(boundary)):
            a, b = (
                ((col, row), (col + 1, row))
                if side == 0
                else ((col + 1, row), (col + 1, row + 1))
                if side == 1
                else ((col + 1, row + 1), (col, row + 1))
                if side == 2
                else ((col, row + 1), (col, row))
            )
            edges.setdefault(a, []).append(b)
    directions = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
    rings: list[list[Point]] = []
    while edges:
        start = min(edges)
        current = start
        direction = None
        ring: list[Point] = []
        while True:
            ring.append(current)
            choices = edges.get(current)
            if not choices:
                raise DatasetValidationError("Broken instance mask boundary")
            if direction is None or len(choices) == 1:
                following = min(choices)
            else:
                # Left turn first joins corner-touching foreground (8-connectivity,
                # matching common instance-mask contour implementations).
                preference = {3: 0, 0: 1, 1: 2, 2: 3}
                following = min(
                    choices,
                    key=lambda p: preference[(directions[(p[0] - current[0], p[1] - current[1])] - direction) % 4],
                )
            direction = directions[(following[0] - current[0], following[1] - current[1])]
            choices.remove(following)
            if not choices:
                del edges[current]
            current = following
            if current == start:
                break
        simplified = [
            point
            for i, point in enumerate(ring)
            if (point[0] - ring[i - 1][0]) * (ring[(i + 1) % len(ring)][1] - point[1])
            != (point[1] - ring[i - 1][1]) * (ring[(i + 1) % len(ring)][0] - point[0])
        ]
        rings.append(simplified)
    outers = [ring for ring in rings if _area(ring) > 0]
    holes = [ring for ring in rings if _area(ring) < 0]
    for hole in holes:
        # A point just inside the hole avoids exact boundary ambiguities.
        a, b = hole[0], hole[1]
        delta = (b[0] - a[0], b[1] - a[1])
        length = max(abs(delta[0]), abs(delta[1]))
        point = ((a[0] + b[0]) / 2 + delta[1] / length * 0.01, (a[1] + b[1]) / 2 - delta[0] / length * 0.01)
        enclosing = [i for i, outer in enumerate(outers) if _contains(outer, point)]
        if not enclosing:
            raise DatasetValidationError("Instance mask hole has no enclosing component")
        index = min(enclosing, key=lambda i: abs(_area(outers[i])))
        outers[index] = _bridge(outers[index], hole)
    mask_height, mask_width = binary.shape
    polygons = tuple(
        _polygon([(min(1.0, x + px / mask_width * width), min(1.0, y + py / mask_height * height)) for px, py in ring])
        for ring in outers
    )
    return polygons, len(holes)


def _instances(value: Any, name: str) -> tuple[HfDamageInstance, ...]:
    kind = _kind(value)
    if kind == "Segmentation":
        raise DatasetValidationError("Semantic Segmentation alone does not preserve damage instance identities/classes")
    if kind in ("Detections", "Polylines"):
        labels = _get(value, "detections" if kind == "Detections" else "polylines", [])
    elif kind in ("Detection", "Polyline"):
        labels = [value]
    else:
        raise DatasetValidationError(f"Unsupported instance annotation type: {kind}")
    result = []
    for label in labels:
        instance_kind = _kind(label)
        if instance_kind == "Detection":
            if _get(label, "iscrowd", False):
                raise DatasetValidationError("Crowd annotation cannot provide individual damage instances")
            mask = _get(label, "mask")
            if mask is None:
                mask_path = _get(label, "mask_path")
                if mask_path:
                    from PIL import Image

                    with Image.open(mask_path) as image:
                        import numpy as np

                        mask = np.array(image)
            if mask is None:
                raise DatasetValidationError(
                    "Detection has no per-instance segmentation mask; bounding-box-only annotations are insufficient"
                )
            polygons, holes = _mask_polygons(mask, _get(label, "bounding_box"))
            result.append(HfDamageInstance(_label(label), polygons, name, "instance_mask", holes))
        elif instance_kind == "Polyline":
            if not _get(label, "closed", False) or not _get(label, "filled", False):
                raise DatasetValidationError("Segmentation polylines must be closed and filled")
            parts = _get(label, "points")
            if not isinstance(parts, (tuple, list)) or not parts:
                raise DatasetValidationError("Segmentation polyline has no polygon components")
            result.append(HfDamageInstance(_label(label), tuple(_polygon(part) for part in parts), name, "polyline"))
        else:
            raise DatasetValidationError(f"Unsupported instance type in {name}: {instance_kind}")
    return tuple(result)


def normalize_fiftyone_dataset(
    dataset: Any, repository: str, revision: str | None, *, annotation_field: str | None = None
) -> HfDamageDataset:
    """Discover schema/usable instance fields from actual loaded FiftyOne data."""
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
            f"Requested annotation field {annotation_field!r} is absent or not a supported label type; schema={field_schema}"
        )
    raw_samples = list(dataset.iter_samples())
    if not raw_samples:
        raise DatasetValidationError("CarDD HF mirror has no samples")
    per_field: dict[str, list[tuple[HfDamageInstance, ...]]] = {}
    summaries = {}
    for name in candidates:
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
    if annotation_field is not None:
        if annotation_field not in per_field:
            raise DatasetValidationError(
                f"Selected annotation field {annotation_field!r} has unusable instance segmentation: {summaries.get(annotation_field)}"
            )
        selected = annotation_field
    elif len(per_field) == 1:
        selected = next(iter(per_field))
    elif len(per_field) > 1:
        raise DatasetValidationError(
            f"Multiple usable annotation fields discovered: {sorted(per_field)}. Select --annotation-field explicitly to avoid duplicate annotations."
        )
    else:
        raise DatasetValidationError(
            f"CarDD HF mirror lacks usable per-instance segmentation masks/polylines. No bbox fallback. Inspected schema={field_schema}; annotation findings={summaries}"
        )
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
            # Portable identity does not depend on the local MongoDB object ID.
            sample_id = "image:" + hashlib.sha256((filepath.name + "\0" + digest).encode()).hexdigest()
        if sample_id in ids:
            raise DatasetValidationError(f"Duplicate stable sample ID: {sample_id}")
        ids.add(sample_id)
        samples.append(HfDamageSample(sample_id, filepath, width, height, instances, digest))
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


def inspect_cardd_hf(dataset: HfDamageDataset) -> dict[str, Any]:
    counts = Counter(instance.label for sample in dataset.samples for instance in sample.instances)
    image_counts = Counter(
        label for sample in dataset.samples for label in {instance.label for instance in sample.instances}
    )
    filenames = Counter(sample.filepath.name.casefold() for sample in dataset.samples)
    paths = Counter(str(sample.filepath.resolve()).casefold() for sample in dataset.samples)
    hashes = Counter(sample.image_sha256 for sample in dataset.samples)
    widths, heights = [s.width for s in dataset.samples], [s.height for s in dataset.samples]
    return {
        "source_name": f"CarDD HF mirror ({dataset.repository})",
        "source_repo": dataset.repository,
        "source_revision": dataset.revision,
        "source_format": dataset.source_format,
        "official_cardd_splits_used": False,
        "sample_count": len(dataset.samples),
        "field_schema": dataset.field_schema,
        "annotation_fields": list(dataset.annotation_fields),
        "annotation_field_summaries": dataset.annotation_field_summaries,
        "damage_classes": list(dataset.classes),
        "labeled_instance_count": sum(counts.values()),
        "instances_per_class": {name: counts[name] for name in dataset.classes},
        "images_per_class": {name: image_counts[name] for name in dataset.classes},
        "images_without_annotations": sum(not sample.instances for sample in dataset.samples),
        "segmentation_usable": bool(counts),
        "segmentation": {
            "instance_masks": sum(i.source_kind == "instance_mask" for s in dataset.samples for i in s.instances),
            "polylines": sum(i.source_kind == "polyline" for s in dataset.samples for i in s.instances),
            "multipart_instances": sum(len(i.polygons) > 1 for s in dataset.samples for i in s.instances),
            "mask_holes": sum(i.mask_hole_count for s in dataset.samples for i in s.instances),
            "bbox_fallback": False,
            "mask_polygon_policy": "trace binary pixel-cell edges with 8-connected foreground; preserve component and hole vertices with retraced hole bridges",
            "representation_limitation": "YOLO has one contour per instance; multipart/hole bridges may add pixels during rasterization",
        },
        "image_dimensions": {
            "width_min": min(widths),
            "width_max": max(widths),
            "height_min": min(heights),
            "height_max": max(heights),
            "distinct_sizes": len({(s.width, s.height) for s in dataset.samples}),
        },
        "duplicate_checks": {
            "duplicate_filenames": sorted(k for k, v in filenames.items() if v > 1),
            "duplicate_filepaths": sorted(k for k, v in paths.items() if v > 1),
            "duplicate_image_sha256": sorted(k for k, v in hashes.items() if v > 1),
            "image_hash_scope": "SHA-256 of full encoded image bytes; detects exact copies even when renamed",
        },
        "sample_id_policy": "source coco_id when available, otherwise filename and full image SHA-256; local FiftyOne/MongoDB IDs are not used",
    }


def _resolve_revision(repository: str, revision: str | None) -> str:
    if revision is not None and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return revision.lower()
    try:
        from huggingface_hub import HfApi

        resolved = HfApi().dataset_info(repository, revision=revision).sha
    except Exception as exc:
        raise DatasetValidationError(f"Cannot resolve Hugging Face source revision for {repository}: {exc}") from exc
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", resolved):
        raise DatasetValidationError("Hugging Face did not return an immutable dataset commit SHA")
    return resolved.lower()


class _PinnedHub:
    """Keep official FiftyOne snapshot calls pinned even on versions dropping SHA."""

    def __init__(self, delegate: Any, revision: str, cache: Path):
        self.delegate, self.revision, self.cache = delegate, revision, cache

    def __getattr__(self, name: str) -> Any:
        target = getattr(self.delegate, name)
        if name not in ("snapshot_download", "hf_hub_download"):
            return target

        def pinned(*args, **kwargs):
            if kwargs.get("revision") not in (None, self.revision):
                raise DatasetValidationError("FiftyOne attempted to download a different HF revision")
            kwargs.update(revision=self.revision, cache_dir=str(self.cache))
            return target(*args, **kwargs)

        return pinned


def _official_load(repository: str, revision: str, cache_dir: Path) -> Any:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FIFTYONE_DATABASE_DIR", str(cache_dir / "fiftyone_db"))
    try:
        import fiftyone as fo
        import fiftyone.utils.huggingface as fo_hf
    except ImportError as exc:
        raise DatasetValidationError(
            "FiftyOne is required for this HF export. Install the project's cardd-hf optional dependencies in a supported Python environment."
        ) from exc
    name = "cardd_hf_" + hashlib.sha256((repository + "@" + revision).encode()).hexdigest()[:20]
    with _LOAD_LOCK:
        if fo.dataset_exists(name):
            dataset = fo.load_dataset(name)
            if dataset.info.get("source_repo") != repository or dataset.info.get("source_revision") != revision:
                raise DatasetValidationError("Existing FiftyOne dataset source metadata disagrees with its pinned name")
            return dataset
        previous_dir, previous_hub = fo.config.default_dataset_dir, fo_hf.hfh
        fo.config.default_dataset_dir = str(cache_dir / "fiftyone" / revision)
        fo_hf.hfh = _PinnedHub(previous_hub, revision, cache_dir / "huggingface")
        try:
            dataset = fo_hf.load_from_hub(repository, revision=revision, name=name, persistent=True, overwrite=False)
            dataset.info.update(
                source_repo=repository,
                source_revision=revision,
                cardd_hf_source_repository=repository,
                cardd_hf_source_revision=revision,
                source_name=f"CarDD HF mirror ({repository})",
            )
            dataset.save()
            return dataset
        except Exception:
            # This name was absent before this invocation. Remove only its
            # incomplete database entry; source/cache files remain resumable.
            if fo.dataset_exists(name):
                fo.delete_dataset(name)
            raise
        finally:
            fo.config.default_dataset_dir, fo_hf.hfh = previous_dir, previous_hub


def load_cardd_hf(
    repository: str | None = None,
    *,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
    annotation_field: str | None = None,
    loader: Callable[..., Any] | None = None,
    revision_resolver: Callable[[str, str | None], str] | None = None,
) -> HfDamageDataset:
    """Load the mirror using official ``fiftyone.utils.huggingface.load_from_hub``.

    Dependencies/network/database are lazy. Immutable revisions isolate caches.
    Injected loaders/resolvers support small offline fixtures without MongoDB.
    """
    repository = repository or os.environ.get("CARDD_HF_DATASET", DEFAULT_HF_REPOSITORY)
    if not isinstance(repository, str) or not repository.strip():
        raise DatasetValidationError("CARDD_HF_DATASET must identify a Hugging Face dataset repository")
    revision = revision if revision is not None else os.environ.get("CARDD_HF_REVISION")
    resolved = (revision_resolver or _resolve_revision)(repository, revision)
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", resolved):
        raise DatasetValidationError("Dataset revision resolver must return a 40-character commit SHA")
    cache = Path(cache_dir).expanduser().resolve() if cache_dir is not None else default_hf_cache_dir()
    try:
        raw = (
            _official_load(repository, resolved.lower(), cache)
            if loader is None
            else loader(repository, revision=resolved.lower())
        )
    except DatasetValidationError:
        raise
    except Exception as exc:
        raise DatasetValidationError(
            f"FiftyOne could not load CarDD HF mirror ({repository}) at {resolved}: {type(exc).__name__}: {exc}"
        ) from exc
    return normalize_fiftyone_dataset(raw, repository, resolved.lower(), annotation_field=annotation_field)
