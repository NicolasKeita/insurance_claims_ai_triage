"""Legacy local COCO inspection/preparation and shared YOLO validation.

No dataset, framework, model or weight is loaded at module import. Inspection is
read-only. The COCO adapter preserves supplied splits and never changes sources.
The default HF mirror workflow uses its own frozen split in cardd_hf modules.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")
PREPARATION_VERSION = "cardd_yolo_segmentation_v1"
DEFAULT_PREPARED_DIR = ROOT / "data" / "external_prepared" / "cardd_yolo"
DEFAULT_MANIFEST_PATH = ROOT / "artifacts" / "vision" / "cardd_dataset_manifest.json"
PREPARED_MANIFEST_NAME = "cardd_dataset_manifest.json"


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


def _integer(value: Any, description: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetValidationError(f"{description} must be an integer")
    if minimum is not None and value < minimum:
        raise DatasetValidationError(f"{description} must be >= {minimum}")
    return value


def _relative_filename(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError("Image file_name must be a nonempty relative filename")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ":" in normalized or any(p in ("", ".", "..") for p in normalized.split("/")):
        raise DatasetValidationError(f"Unsafe relative filename: {value!r}")
    return path.as_posix()


def _contained_file(directory: Path, filename: str) -> Path:
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise DatasetValidationError(f"Image escapes its split directory: {filename}")
    if not path.is_file():
        raise DatasetValidationError(f"Image file is missing: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Cannot read JSON annotations: {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise DatasetValidationError(f"Annotations must contain a JSON object: {path}")
    return document


def read_category_mapping(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Map actual COCO category IDs, including noncontiguous IDs, to YOLO IDs."""
    categories = document.get("categories")
    if not isinstance(categories, list) or not categories:
        raise DatasetValidationError("COCO categories must be a nonempty list")
    names: set[str] = set()
    ids: set[int] = set()
    result = []
    for category in categories:
        if not isinstance(category, dict):
            raise DatasetValidationError("Each COCO category must be an object")
        category_id = _integer(category.get("id"), "COCO category ID", minimum=0)
        name = category.get("name")
        if not isinstance(name, str) or not name.strip():
            raise DatasetValidationError("COCO category names must be nonempty strings")
        if category_id in ids or name.casefold().strip() in names:
            raise DatasetValidationError("Duplicate COCO category ID or name")
        ids.add(category_id)
        names.add(name.casefold().strip())
        result.append({"category_id": category_id, "name": name})
    result.sort(key=lambda item: item["category_id"])
    return [dict(item, yolo_class_id=index) for index, item in enumerate(result)]


def _discover_path(root: Path, candidates: list[str], kind: str, split: str) -> Path:
    found = [root / candidate for candidate in candidates if (root / candidate).exists()]
    if len(found) != 1:
        detail = "none found" if not found else "ambiguous: " + ", ".join(str(p) for p in found)
        raise DatasetValidationError(
            f"CarDD {split} {kind}: {detail}. Expected one of {candidates}; use explicit path mappings for a custom layout."
        )
    return found[0].resolve()


def discover_cardd_dataset(
    dataset_dir: Path | str | None = None,
    *,
    annotation_files: dict[str, Path | str] | None = None,
    image_directories: dict[str, Path | str] | None = None,
) -> CarddDataset:
    """Read official CarDD_COCO layout, require all splits, and reject leakage."""
    configured = dataset_dir if dataset_dir is not None else os.environ.get("CARDD_DATASET_DIR")
    if not configured:
        raise DatasetValidationError(
            "CarDD dataset unavailable. Obtain CarDD manually from its official site under its conditions, then set CARDD_DATASET_DIR to the CarDD_COCO directory. No automatic download is performed."
        )
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise DatasetValidationError(
            f"CARDD_DATASET_DIR does not exist or is not a directory: {root}. Set it to the manually obtained CarDD_COCO directory."
        )
    for mapping in (annotation_files, image_directories):
        if mapping is not None and set(mapping) != set(SPLITS):
            raise DatasetValidationError("Explicit split paths must define exactly train, val and test")
    splits = {}
    category_mapping = None
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for split in SPLITS:
        if annotation_files is None:
            annotation_path = _discover_path(
                root,
                [f"annotations/instances_{split}2017.json", f"annotations/instances_{split}.json"],
                "annotations",
                split,
            )
        else:
            annotation_path = (root / annotation_files[split]).resolve()
        if not annotation_path.is_file():
            raise DatasetValidationError(f"CarDD {split} annotation file is missing: {annotation_path}")
        if image_directories is None:
            image_dir = _discover_path(
                root, [f"{split}2017", f"images/{split}2017", f"images/{split}", split], "image directory", split
            )
        else:
            image_dir = (root / image_directories[split]).resolve()
        if not image_dir.is_dir():
            raise DatasetValidationError(f"CarDD {split} image directory is missing: {image_dir}")
        document = _read_json(annotation_path)
        mapping = read_category_mapping(document)
        if category_mapping is None:
            category_mapping = mapping
        elif category_mapping != mapping:
            raise DatasetValidationError(f"COCO category IDs/names differ between splits: {split}")
        images = document.get("images")
        annotations = document.get("annotations")
        if not isinstance(images, list) or not images or not isinstance(annotations, list):
            raise DatasetValidationError(f"{split}: COCO images must be nonempty and annotations must be a list")
        split_ids, split_names, label_names = set(), set(), set()
        for image in images:
            if not isinstance(image, dict):
                raise DatasetValidationError(f"{split}: COCO image must be an object")
            image_id = _integer(image.get("id"), f"{split} image ID", minimum=0)
            filename = _relative_filename(image.get("file_name"))
            _integer(image.get("width"), f"{split} image width", minimum=1)
            _integer(image.get("height"), f"{split} image height", minimum=1)
            label_name = str(PurePosixPath(filename).with_suffix(".txt")).casefold()
            if image_id in split_ids or filename.casefold() in split_names or label_name in label_names:
                raise DatasetValidationError(f"{split}: duplicate image ID, filename or label filename")
            _contained_file(image_dir, filename)
            split_ids.add(image_id)
            split_names.add(filename.casefold())
            label_names.add(label_name)
        if split_ids & seen_ids or split_names & seen_names:
            raise DatasetValidationError(f"Split leakage: {split} overlaps previous splits by image ID or filename")
        seen_ids.update(split_ids)
        seen_names.update(split_names)
        annotation_ids = set()
        category_ids = {item["category_id"] for item in mapping}
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise DatasetValidationError(f"{split}: COCO annotation must be an object")
            annotation_id = _integer(annotation.get("id"), f"{split} annotation ID", minimum=0)
            if annotation_id in annotation_ids:
                raise DatasetValidationError(f"{split}: duplicate annotation ID")
            annotation_ids.add(annotation_id)
            if _integer(annotation.get("image_id"), "Annotation image ID") not in split_ids:
                raise DatasetValidationError(f"{split}: annotation references an unknown image ID")
            if _integer(annotation.get("category_id"), "Annotation category ID") not in category_ids:
                raise DatasetValidationError(f"{split}: annotation references an unknown category ID")
        splits[split] = CocoSplit(split, annotation_path, image_dir, document)
    return CarddDataset(root, splits, category_mapping or [])


def annotation_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_cardd_dataset(dataset: CarddDataset) -> dict[str, Any]:
    """Portable inspection metadata. Does not write a manifest or any other file."""
    mapping = dataset.category_mapping
    id_to_name = {item["category_id"]: item["name"] for item in mapping}
    split_counts = {}
    for name, split in dataset.splits.items():
        counts: Counter[str] = Counter()
        images_per_category: dict[str, set[int]] = {item["name"]: set() for item in mapping}
        annotated = set()
        for annotation in split.document["annotations"]:
            category_name = id_to_name[annotation["category_id"]]
            counts[category_name] += 1
            images_per_category[category_name].add(annotation["image_id"])
            annotated.add(annotation["image_id"])
        split_counts[name] = {
            "image_count": len(split.document["images"]),
            "annotation_count": len(split.document["annotations"]),
            "annotations_per_category": {item["name"]: counts[item["name"]] for item in mapping},
            "images_per_category": {key: len(value) for key, value in images_per_category.items()},
            "images_without_annotations": len(split.document["images"]) - len(annotated),
        }
    return {
        "category_mapping": [dict(item) for item in mapping],
        "splits": split_counts,
        "annotation_hashes": {name: annotation_hash(split.annotation_path) for name, split in dataset.splits.items()},
        "split_overlap_check": "passed: image IDs and case-insensitive filenames are disjoint",
    }


def build_dataset_manifest(dataset: CarddDataset) -> dict[str, Any]:
    result = inspect_cardd_dataset(dataset)
    result["preparation_version"] = PREPARATION_VERSION
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    stable = {
        "annotation_hashes": result["annotation_hashes"],
        "category_mapping": result["category_mapping"],
        "preparation_version": PREPARATION_VERSION,
    }
    result["dataset_fingerprint"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def _polygon(annotation: dict[str, Any], width: int, height: int) -> tuple[list[float] | None, str | None]:
    if annotation.get("iscrowd", 0):
        return None, "crowd"
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, dict):
        return None, "rle"
    if not segmentation:
        return None, "missing_segmentation"
    if not isinstance(segmentation, list) or any(not isinstance(part, list) for part in segmentation):
        return None, "invalid_polygon"
    parts = []
    areas = []
    for values in segmentation:
        if len(values) < 6 or len(values) % 2:
            return None, "invalid_polygon"
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in values
        ):
            return None, "invalid_coordinates"
        points = [(float(values[i]), float(values[i + 1])) for i in range(0, len(values), 2)]
        if any(not 0 <= x <= width or not 0 <= y <= height for x, y in points):
            return None, "invalid_coordinates"
        if points[0] == points[-1]:
            points.pop()
        area = (
            sum(
                x * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * y
                for i, (x, y) in enumerate(points)
            )
            if points
            else 0
        )
        if len(set(points)) < 3 or abs(area) < 1e-9:
            return None, "degenerate_polygon"
        parts.append(points)
        areas.append(area)
    points = parts[0]
    # YOLO stores one polygon per instance. Preserve every component vertex and
    # retrace each connector so it adds zero polygon area. Matching winding keeps
    # component areas additive. This cannot preserve COCO disconnected topology
    # exactly after rasterization; the manifest records the representation limit.
    for component, area in zip(parts[1:], areas[1:]):
        if area * areas[0] < 0:
            component = component[::-1]
        _, left, right = min(
            ((x - a) ** 2 + (y - b) ** 2, i, j) for i, (x, y) in enumerate(points) for j, (a, b) in enumerate(component)
        )
        rotated = component[right:] + component[:right]
        points = points[: left + 1] + rotated + [rotated[0], points[left]] + points[left + 1 :]
    return [value for x, y in points for value in (x / width, y / height)], None


def validate_segmentation_label(
    path: Path, *, class_count: int, expected_count: int | None = None
) -> list[tuple[int, list[float]]]:
    """Validate YOLO polygon labels; a five-field bounding box is never accepted."""
    if not path.is_file():
        raise DatasetValidationError(f"Missing prepared label: {path}")
    rows = []
    seen_rows = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise DatasetValidationError(f"Blank segmentation row at {path}:{line_number}")
        fields = line.split()
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except (ValueError, IndexError) as exc:
            raise DatasetValidationError(f"Invalid segmentation row at {path}:{line_number}") from exc
        if not 0 <= class_id < class_count or len(coordinates) < 6 or len(coordinates) % 2:
            raise DatasetValidationError(f"Invalid class ID or polygon vertex count at {path}:{line_number}")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates):
            raise DatasetValidationError(
                f"Polygon coordinates must be finite and normalized to [0,1] at {path}:{line_number}"
            )
        polygon, reason = _polygon({"segmentation": [coordinates]}, 1, 1)
        if reason or polygon is None:
            raise DatasetValidationError(f"Invalid polygon ({reason}) at {path}:{line_number}")
        key = (class_id, *coordinates)
        if key in seen_rows:
            raise DatasetValidationError(f"Duplicate segmentation row would be removed by YOLO at {path}:{line_number}")
        seen_rows.add(key)
        rows.append((class_id, coordinates))
    if expected_count is not None and len(rows) != expected_count:
        raise DatasetValidationError(f"Unexpected label count for {path}: expected {expected_count}, found {len(rows)}")
    return rows


def _check_image(path: Path, width: int, height: int) -> None:
    from PIL import Image, UnidentifiedImageError

    if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
        raise DatasetValidationError(f"Image format unsupported by this preparation pipeline: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != (width, height):
                raise DatasetValidationError(f"Image dimensions disagree with COCO/prepared metadata: {path}")
            if image.getexif().get(274, 1) not in (None, 1):
                raise DatasetValidationError(
                    f"EXIF orientation would change annotation coordinates: {path}; source needs correctly oriented COCO annotations"
                )
            image.load()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise DatasetValidationError(f"Image is not decodable: {path}: {exc}") from exc


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def prepare_cardd_yolo(
    dataset: CarddDataset,
    output_dir: Path = DEFAULT_PREPARED_DIR,
    *,
    manifest_path: Path | None = DEFAULT_MANIFEST_PATH,
    unsupported: str = "error",
    converter: str = "auto",
) -> dict[str, Any]:
    """Prepare atomically, with explicit unsupported mask policy and validation.

    ``error`` rejects crowd, RLE, missing, malformed and duplicate segmentations.
    ``skip`` records every excluded reason and count. Neither creates fake masks.
    Multipart polygons preserve component vertices with retraced bridges because
    YOLO supports only one polygon per instance; this limitation is recorded.
    ``auto`` uses the official converter if available and output-compatible, then
    records a native polygon-only fallback when it is absent or incompatible.
    """
    if unsupported not in ("error", "skip") or converter not in ("auto", "native", "ultralytics"):
        raise DatasetValidationError("Invalid unsupported annotation policy or converter")
    output = Path(output_dir).resolve()
    source_locations = [
        dataset.root,
        *(split.image_dir for split in dataset.splits.values()),
        *(split.annotation_path.parent for split in dataset.splits.values()),
    ]
    if any(
        output == source or output.is_relative_to(source) or source.is_relative_to(output)
        for source in source_locations
    ):
        raise DatasetValidationError("Prepared output must be outside and separate from the source dataset")
    if manifest_path is not None and any(
        Path(manifest_path).resolve().is_relative_to(source) for source in source_locations
    ):
        raise DatasetValidationError("Generated manifest must be outside the source dataset")
    if output.exists():
        raise DatasetValidationError(
            f"Prepared output already exists: {output}. Choose a new versioned --output-dir; existing data is never overwritten."
        )
    from vision.taxonomy import map_category_name

    manifest = build_dataset_manifest(dataset)
    for category in manifest["category_mapping"]:
        try:
            category["damage_type"] = map_category_name(category["name"]).value
        except ValueError as exc:
            raise DatasetValidationError(f"Unsupported domain taxonomy category: {category['name']}") from exc
    # Two different source names must never accidentally collapse into one class.
    if len({item["damage_type"] for item in manifest["category_mapping"]}) != len(manifest["category_mapping"]):
        raise DatasetValidationError("COCO categories collapse to duplicate domain damage types")
    class_ids = {item["category_id"]: item["yolo_class_id"] for item in manifest["category_mapping"]}
    expected: dict[str, dict[str, list[tuple[int, list[float]]]]] = {}
    documents = {}
    manifest["prepared_files"] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cardd-preparation-", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
        stage = temporary_root / "dataset"
        stage.mkdir()
        for split_name, split in dataset.splits.items():
            images = {image["id"]: image for image in split.document["images"]}
            rows = {image_id: [] for image_id in images}
            normalized_annotations = []
            skipped: Counter[str] = Counter()
            skipped_by_class: Counter[str] = Counter()
            stitched_multipart_count = 0
            class_names = {item["category_id"]: item["name"] for item in manifest["category_mapping"]}
            for annotation in split.document["annotations"]:
                image = images[annotation["image_id"]]
                polygon, reason = _polygon(annotation, image["width"], image["height"])
                class_id = class_ids[annotation["category_id"]]
                if reason is None and (class_id, polygon) in rows[annotation["image_id"]]:
                    reason = "duplicate_polygon"
                if reason:
                    skipped[reason] += 1
                    skipped_by_class[class_names[annotation["category_id"]]] += 1
                    if unsupported == "error":
                        raise DatasetValidationError(
                            f"{split_name} annotation {annotation['id']}: unusable segmentation ({reason}). Fix the source annotations or explicitly choose --unsupported skip to measure and exclude it; no bounding-box fallback."
                        )
                    continue
                stitched_multipart_count += len(annotation.get("segmentation", [])) > 1
                rows[annotation["image_id"]].append((class_id, polygon or []))
                xs = (polygon or [])[::2]
                ys = (polygon or [])[1::2]
                normalized_annotations.append(
                    dict(
                        annotation,
                        category_id=class_id + 1,
                        iscrowd=0,
                        segmentation=[
                            [
                                value * (image["width"] if i % 2 == 0 else image["height"])
                                for i, value in enumerate(polygon or [])
                            ]
                        ],
                        bbox=[
                            min(xs) * image["width"],
                            min(ys) * image["height"],
                            (max(xs) - min(xs)) * image["width"],
                            (max(ys) - min(ys)) * image["height"],
                        ],
                    )
                )
            expected[split_name] = {}
            prepared_files = []
            for image_id, image in images.items():
                filename = _relative_filename(image["file_name"])
                source = _contained_file(split.image_dir, filename)
                _check_image(source, image["width"], image["height"])
                image_path = stage / "images" / split_name / filename
                image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, image_path)
                label_relative = (
                    PurePosixPath("labels") / split_name / PurePosixPath(filename).with_suffix(".txt")
                ).as_posix()
                label_path = stage / label_relative
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(
                    "".join(
                        str(class_id) + " " + " ".join(format(value, ".12g") for value in polygon) + "\n"
                        for class_id, polygon in rows[image_id]
                    ),
                    encoding="utf-8",
                )
                expected[split_name][filename] = rows[image_id]
                prepared_files.append(
                    {
                        "image": (PurePosixPath("images") / split_name / filename).as_posix(),
                        "label": label_relative,
                        "image_id": image_id,
                        "width": image["width"],
                        "height": image["height"],
                        "row_count": len(rows[image_id]),
                        "class_counts": dict(Counter(str(row[0]) for row in rows[image_id])),
                    }
                )
            manifest["prepared_files"][split_name] = prepared_files
            counts = manifest["splits"][split_name]
            counts.update(
                converted_annotation_count=sum(len(value) for value in rows.values()),
                skipped_annotations=dict(skipped),
                skipped_annotations_per_category=dict(skipped_by_class),
                stitched_multipart_annotations=stitched_multipart_count,
                label_count=len(images),
                images_without_converted_annotations=sum(not value for value in rows.values()),
            )
            converted_counts = Counter(class_id for values in rows.values() for class_id, _ in values)
            converted_images = Counter(class_id for values in rows.values() for class_id in {row[0] for row in values})
            counts["converted_annotations_per_category"] = {
                item["name"]: converted_counts[item["yolo_class_id"]] for item in manifest["category_mapping"]
            }
            counts["converted_images_per_category"] = {
                item["name"]: converted_images[item["yolo_class_id"]] for item in manifest["category_mapping"]
            }
            documents[split_name] = {
                "images": list(images.values()),
                "annotations": normalized_annotations,
                "categories": [
                    {"id": item["yolo_class_id"] + 1, "name": item["name"]} for item in manifest["category_mapping"]
                ],
            }
        conversion = {
            "backend": "native_polygon",
            "unsupported_policy": unsupported,
            "segmentation_enabled": True,
            "bbox_fallback": False,
            "multipart_policy": "preserve component vertices using retraced nearest-vertex bridges",
            "multipart_limitation": "YOLO uses one polygon per instance; disconnected topology and holes cannot be represented exactly, and rasterization may include bridge pixels",
            "use_segments": True,
            "cls91to80": False,
        }
        if converter != "native":
            try:
                _try_ultralytics_conversion(stage, documents, expected, len(class_ids))
                conversion["backend"] = "ultralytics.convert_coco"
            except Exception as exc:
                if converter == "ultralytics":
                    raise DatasetValidationError(
                        f"Official COCO segmentation conversion failed validation: {exc}"
                    ) from exc
                conversion["fallback_reason"] = (
                    f"{type(exc).__name__}: official converter unavailable, incompatible, or failed strict polygon validation"
                )
        manifest["conversion"] = conversion
        for files in manifest["prepared_files"].values():
            for entry in files:
                entry["label_sha256"] = annotation_hash(stage / entry["label"])
        # JSON is valid YAML and avoids a YAML dependency for preparation.
        dataset_yaml = {
            "path": str(stage),
            **{name: f"images/{name}" for name in SPLITS},
            "names": [item["name"] for item in manifest["category_mapping"]],
            "nc": len(class_ids),
        }
        _write_json(stage / "dataset.yaml", dataset_yaml)
        _write_json(stage / PREPARED_MANIFEST_NAME, manifest)
        load_prepared_dataset(stage / "dataset.yaml")
        # Converter scratch data is not part of the prepared dataset.
        for scratch in (stage / "converter_annotations", stage / "converter_output"):
            if scratch.exists():
                shutil.rmtree(scratch)
        # Ultralytics resolves relative dataset paths against its datasets_dir,
        # not the YAML directory. Only portable manifest metadata excludes root.
        dataset_yaml["path"] = str(output)
        _write_json(stage / "dataset.yaml", dataset_yaml)
        stage.rename(output)
    if manifest_path is not None:
        _write_json(Path(manifest_path), manifest)
    return manifest


def load_prepared_dataset(dataset_yaml: Path) -> dict[str, Any]:
    """Verify dataset/manifest, actual images, labels, counts and split isolation."""
    yaml_path = Path(dataset_yaml).resolve()
    if not yaml_path.is_file():
        raise DatasetValidationError(
            f"Prepared dataset YAML is missing: {yaml_path}. Run scripts/prepare_cardd_yolo.py first."
        )
    try:
        document = json.loads(yaml_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        import yaml

        document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise DatasetValidationError("Prepared dataset YAML must contain a mapping")
    dataset_root = (yaml_path.parent / document.get("path", ".")).resolve()
    if dataset_root != yaml_path.parent:
        raise DatasetValidationError("Prepared dataset path must resolve to the YAML's parent directory")
    manifest_path = dataset_root / PREPARED_MANIFEST_NAME
    if not manifest_path.is_file():
        raise DatasetValidationError(f"Prepared dataset manifest missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    from vision.cardd_hf_preparation import HF_PREPARATION_VERSION

    is_hf_mirror = manifest.get("preparation_version") == HF_PREPARATION_VERSION
    if manifest.get("preparation_version") != PREPARATION_VERSION and not is_hf_mirror:
        raise DatasetValidationError("Unsupported prepared dataset manifest version")
    if is_hf_mirror:
        from vision.cardd_hf_preparation import validate_hf_prepared_manifest

        validate_hf_prepared_manifest(manifest)
    mapping = manifest.get("category_mapping")
    if not isinstance(mapping, list) or not mapping:
        raise DatasetValidationError("Prepared category mapping must be nonempty")
    try:
        from vision.taxonomy import map_category_name

        valid_mapping = read_category_mapping(
            {"categories": [{"id": item["category_id"], "name": item["name"]} for item in mapping]}
        )
        expected_mapping = [dict(item, damage_type=map_category_name(item["name"]).value) for item in valid_mapping]
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetValidationError(f"Invalid prepared category mapping: {exc}") from exc
    if mapping != expected_mapping:
        raise DatasetValidationError("Prepared category mapping is not contiguous or has inconsistent domain types")
    hashes = manifest.get("annotation_hashes", {})
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(SPLITS)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes.values()
        )
    ):
        raise DatasetValidationError("Prepared manifest must contain SHA-256 annotation hashes for all splits")
    if is_hf_mirror:
        from vision.cardd_hf_preparation import hf_prepared_fingerprint

        fingerprint = hf_prepared_fingerprint(manifest, valid_mapping)
    else:
        stable = {
            "annotation_hashes": hashes,
            "category_mapping": valid_mapping,
            "preparation_version": PREPARATION_VERSION,
        }
        fingerprint = hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("dataset_fingerprint") != fingerprint:
        raise DatasetValidationError("Prepared manifest fingerprint disagrees with categories/annotation hashes")
    if document.get("names") != [item["name"] for item in mapping] or document.get("nc") != len(mapping):
        raise DatasetValidationError("dataset.yaml class names/count disagree with the manifest")
    if set(manifest.get("splits", {})) != set(SPLITS) or set(manifest.get("prepared_files", {})) != set(SPLITS):
        raise DatasetValidationError("Prepared dataset must contain independent train, val and test splits")
    seen_ids, seen_names = set(), set()
    for split in SPLITS:
        if document.get(split) != f"images/{split}":
            raise DatasetValidationError(f"Prepared {split} split path must be images/{split}")
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise DatasetValidationError(f"Prepared {split} image/label directory missing")
        entries = manifest["prepared_files"][split]
        if not isinstance(entries, list) or not entries:
            raise DatasetValidationError(f"Prepared {split} image metadata must be nonempty")
        expected_images, expected_labels, split_names, split_ids = set(), set(), set(), set()
        row_count = 0
        actual_class_counts: Counter[int] = Counter()
        actual_class_images: Counter[int] = Counter()
        for entry in entries:
            try:
                image_name = _relative_filename(entry["image"])
                label_name = _relative_filename(entry["label"])
                image_id = _integer(entry["image_id"], "Prepared image ID", minimum=0)
                width = _integer(entry["width"], "Prepared image width", minimum=1)
                height = _integer(entry["height"], "Prepared image height", minimum=1)
                count = _integer(entry["row_count"], "Prepared label count", minimum=0)
            except (KeyError, TypeError) as exc:
                raise DatasetValidationError("Invalid prepared image/label metadata") from exc
            if (
                not image_name.startswith(f"images/{split}/")
                or not label_name.startswith(f"labels/{split}/")
                or PurePosixPath(image_name).with_suffix(".txt").parts[2:] != PurePosixPath(label_name).parts[2:]
            ):
                raise DatasetValidationError("Prepared label does not reference a corresponding split image")
            image_path = _contained_file(dataset_root, image_name)
            label_path = _contained_file(dataset_root, label_name)
            _check_image(image_path, width, height)
            if is_hf_mirror and annotation_hash(image_path) != entry.get("image_sha256"):
                raise DatasetValidationError(f"Prepared HF image content changed: {image_path}")
            rows = validate_segmentation_label(label_path, class_count=len(mapping), expected_count=count)
            if dict(Counter(str(row[0]) for row in rows)) != entry.get("class_counts") or annotation_hash(
                label_path
            ) != entry.get("label_sha256"):
                raise DatasetValidationError(
                    f"Prepared labels changed or class counts disagree with manifest: {label_path}"
                )
            filename = PurePosixPath(image_name).relative_to(PurePosixPath("images") / split).as_posix().casefold()
            if (
                image_id in split_ids
                or filename in split_names
                or image_path in expected_images
                or label_path in expected_labels
            ):
                raise DatasetValidationError(f"Duplicate prepared image or label metadata in {split}")
            split_ids.add(image_id)
            split_names.add(filename)
            expected_images.add(image_path)
            expected_labels.add(label_path)
            row_count += len(rows)
            actual_class_counts.update(row[0] for row in rows)
            actual_class_images.update({row[0] for row in rows})
        if seen_ids & split_ids or seen_names & split_names:
            raise DatasetValidationError(f"Prepared split leakage in {split}")
        seen_ids.update(split_ids)
        seen_names.update(split_names)
        actual_images = {path.resolve() for path in image_dir.rglob("*") if path.is_file()}
        actual_labels = {path.resolve() for path in label_dir.rglob("*") if path.is_file()}
        if actual_images != expected_images or actual_labels != expected_labels:
            raise DatasetValidationError(f"Unexpected or missing files in prepared {split} split")
        counts = manifest["splits"][split]
        if (
            counts.get("image_count") != len(entries)
            or counts.get("label_count") != len(entries)
            or counts.get("converted_annotation_count") != row_count
        ):
            raise DatasetValidationError(f"Prepared {split} counts disagree with actual labels/images")
        skipped = counts.get("skipped_annotations", {})
        if (
            not isinstance(skipped, dict)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in skipped.values())
            or counts.get("annotation_count") != row_count + sum(skipped.values())
        ):
            raise DatasetValidationError(f"Prepared {split} annotation/skip counts are inconsistent")
        if counts.get("converted_annotations_per_category") != {
            item["name"]: actual_class_counts[item["yolo_class_id"]] for item in mapping
        } or counts.get("converted_images_per_category") != {
            item["name"]: actual_class_images[item["yolo_class_id"]] for item in mapping
        }:
            raise DatasetValidationError(f"Prepared {split} category distribution disagrees with actual labels")
    return manifest
