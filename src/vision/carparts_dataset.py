"""Official Carparts-Seg preparation and read-only inspection; no import downloads."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from vision.dataset import DatasetValidationError, SPLITS, validate_segmentation_label
from vision.training import REPO_ROOT, file_sha256, write_json

PREPARATION_VERSION = "carparts_official_yolo_segmentation_v1"
OFFICIAL_SOURCE = "https://docs.ultralytics.com/datasets/segment/carparts-seg/"
OFFICIAL_ARCHIVE = "https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/external/carparts-seg"
DEFAULT_DATASET_YAML = DEFAULT_DATASET_ROOT / "dataset.yaml"
DEFAULT_MANIFEST = REPO_ROOT / "artifacts/vision/carparts_dataset_manifest.json"
MANIFEST_NAME = "carparts_dataset_manifest.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _read_yaml(path: Path) -> dict[str, Any]:
    # PyYAML is a small transitive vision dependency, imported only on inspection.
    import yaml
    document = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise DatasetValidationError(f"Dataset YAML must contain an object: {path}")
    return document


def _class_mapping(document: dict[str, Any]) -> list[dict[str, Any]]:
    names = document.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise DatasetValidationError("Dataset class names must be a nonempty list or mapping")
    try:
        names = {int(key): value for key, value in names.items()}
    except (ValueError, TypeError) as exc:
        raise DatasetValidationError("Dataset class IDs must be integers") from exc
    if sorted(names) != list(range(len(names))):
        raise DatasetValidationError("Dataset class IDs must be contiguous from zero")
    if any(not isinstance(name, str) or not name.strip() for name in names.values()) or len(set(names.values())) != len(names):
        raise DatasetValidationError("Dataset class names must be unique nonempty strings")
    if document.get("nc", len(names)) != len(names):
        raise DatasetValidationError("Dataset nc disagrees with class names")
    return [{"yolo_class_id": index, "category_id": index, "name": names[index]} for index in range(len(names))]


def inspect_carparts_dataset(dataset_yaml: Path = DEFAULT_DATASET_YAML) -> dict[str, Any]:
    """Validate all actual images/YOLO polygons and count the supplied splits.

    This never invokes an Ultralytics downloader, even for a missing directory.
    Label hashes are aggregated per split; images are decoded but not hashed.
    """
    from PIL import Image
    dataset_yaml = Path(dataset_yaml).resolve()
    document = _read_yaml(dataset_yaml)
    mapping = _class_mapping(document)
    root = Path(document.get("path") or dataset_yaml.parent)
    if not root.is_absolute():
        root = dataset_yaml.parent / root
    root = root.resolve()
    report: dict[str, Any] = {
        "dataset_name": "Carparts-Seg", "dataset_root": str(root),
        "dataset_yaml": str(dataset_yaml), "dataset_yaml_sha256": file_sha256(dataset_yaml),
        "category_mapping": mapping, "splits": {}, "prepared_files": {},
        "annotation_hashes": {}, "invalid_labels": [], "invalid_images": [],
        "missing_labels": [], "orphan_labels": [], "split_path_errors": [],
        "official_splits_used": True, "preparation_version": PREPARATION_VERSION,
    }
    seen_images: set[Path] = set()
    for split in SPLITS:
        expected_path = f"images/{split}"
        if document.get(split) != expected_path:
            raise DatasetValidationError(f"Carparts-Seg must retain its supplied {expected_path} split")
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            report["split_path_errors"].append(f"Missing {split} image/label directories")
        images = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        instances: Counter[int] = Counter()
        class_images: Counter[int] = Counter()
        digest = hashlib.sha256()
        entries = []
        expected_labels: set[Path] = set()
        empty_images = []
        for image in images:
            resolved = image.resolve()
            if not resolved.is_relative_to(root) or resolved in seen_images:
                raise DatasetValidationError(f"Image escapes dataset root or occurs in multiple splits: {image}")
            seen_images.add(resolved)
            label = label_dir / image.relative_to(image_dir).with_suffix(".txt")
            expected_labels.add(label.resolve())
            width = height = None
            try:
                with Image.open(image) as decoded:
                    width, height = decoded.size
                    decoded.verify()
                if width <= 0 or height <= 0:
                    raise ValueError("Image has invalid dimensions")
            except (OSError, ValueError) as exc:
                report["invalid_images"].append({"image": str(image.relative_to(root)), "error": str(exc)})
            rows = []
            if not label.is_file():
                report["missing_labels"].append(str(label.relative_to(root)))
            else:
                try:
                    rows = validate_segmentation_label(label, class_count=len(mapping))
                except DatasetValidationError as exc:
                    report["invalid_labels"].append({"label": str(label.relative_to(root)), "error": str(exc)})
                digest.update(label.relative_to(root).as_posix().encode())
                digest.update(bytes.fromhex(file_sha256(label)))
            if not rows:
                empty_images.append(str(image.relative_to(root)))
            instances.update(row[0] for row in rows)
            class_images.update({row[0] for row in rows})
            entries.append({"image": image.relative_to(root).as_posix(), "label": label.relative_to(root).as_posix(),
                            "width": width, "height": height, "row_count": len(rows)})
        labels = {path.resolve() for path in label_dir.rglob("*.txt")}
        report["orphan_labels"].extend(str(path.relative_to(root)) for path in sorted(labels - expected_labels))
        report["splits"][split] = {
            "image_count": len(images), "label_count": len(labels), "annotation_count": sum(instances.values()),
            "images_without_annotations": empty_images, "images_without_annotation_count": len(empty_images),
            "annotations_per_category": {item["name"]: instances[item["yolo_class_id"]] for item in mapping},
            "images_per_category": {item["name"]: class_images[item["yolo_class_id"]] for item in mapping},
        }
        report["prepared_files"][split] = entries
        report["annotation_hashes"][split] = digest.hexdigest()
    report["actual_sample_count"] = sum(value["image_count"] for value in report["splits"].values())
    source_names = {split: {Path(entry["image"]).name.split(".rf.")[0]
                            for entry in report["prepared_files"][split]} for split in SPLITS}
    report["source_filename_overlap_audit"] = {
        "method": "Filename prefix before .rf.; repeated source identifiers only, no image hashes or pixel comparison.",
        "unique_source_identifiers": {split: len(values) for split, values in source_names.items()},
        "pairwise_shared_identifiers": {f"{left}_{right}": len(source_names[left] & source_names[right])
                                        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))},
        "limitation": "Supplied splits are preserved. Shared filename identifiers suggest related/augmented source images across splits; they do not prove identical pixels or independent unseen-photo generalization.",
    }
    report["valid"] = (all(value["image_count"] > 0 for value in report["splits"].values())
                       and not any(report[key] for key in ("invalid_labels", "invalid_images", "missing_labels", "orphan_labels", "split_path_errors")))
    return report


def _require_valid(report: dict[str, Any]) -> None:
    if not report["valid"]:
        counts = {key: len(report[key]) for key in ("invalid_labels", "invalid_images", "missing_labels", "orphan_labels", "split_path_errors")}
        raise DatasetValidationError(f"Carparts-Seg inspection failed: {counts}; inspect the saved report before training")


def prepare_carparts_dataset(
    *, dataset_root: Path = DEFAULT_DATASET_ROOT, manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Explicit official download followed by real inspection and reproducibility manifest."""
    import ultralytics
    from ultralytics.utils import ROOT as ULTRALYTICS_ROOT
    from ultralytics.utils.downloads import safe_download

    dataset_root = Path(dataset_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    official_yaml = ULTRALYTICS_ROOT / "cfg/datasets/carparts-seg.yaml"
    official_document = _read_yaml(official_yaml)
    if official_document.get("download") != OFFICIAL_ARCHIVE:
        raise DatasetValidationError("Installed official Carparts-Seg YAML changed its download URL; review before downloading")
    # Existing folders are inspected below, never considered proof of success.
    if not dataset_root.is_dir():
        extracted = Path(safe_download(OFFICIAL_ARCHIVE, dir=dataset_root.parent, unzip=True, delete=False))
        if extracted.resolve() != dataset_root:
            raise DatasetValidationError(f"Official archive extracted to an unexpected root: {extracted}")
    candidates = [dataset_root / name for name in ("data.yaml", "carparts-seg.yaml") if (dataset_root / name).is_file()]
    source_yaml = candidates[0] if candidates else official_yaml
    source_document = _read_yaml(source_yaml)
    mapping = _class_mapping(source_document)
    if mapping != _class_mapping(official_document):
        raise DatasetValidationError("Archive classes differ from the installed official dataset YAML")
    copied_source = dataset_root / "source_dataset.yaml"
    copied_source.write_bytes(source_yaml.read_bytes())
    dataset_yaml = dataset_root / "dataset.yaml"
    normalized = {"path": str(dataset_root), **{split: f"images/{split}" for split in SPLITS},
                  "names": {item["yolo_class_id"]: item["name"] for item in mapping}, "nc": len(mapping)}
    write_json(dataset_yaml, normalized)  # JSON is valid YAML; no download directive.
    inspection = inspect_carparts_dataset(dataset_yaml)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path.with_name("carparts_dataset_inspection.json"), inspection)
    _require_valid(inspection)
    license_files = sorted(path for path in dataset_root.rglob("*") if path.is_file()
                           and ("license" in path.name.lower() or "readme" in path.name.lower()))
    manifest = inspection | {
        "schema_version": 1, "source": OFFICIAL_SOURCE, "source_name": "Ultralytics official Carparts-Seg",
        "source_repository": OFFICIAL_ARCHIVE,
        "license": {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/",
                    "evidence": "CC BY 4.0 badge on Ultralytics dataset documentation",
                    "attribution": "Gianmarco Russo; car-seg dataset on Roboflow Universe (2023)",
                    "archive_metadata_files": [path.relative_to(dataset_root).as_posix() for path in license_files]},
        "ultralytics_version": str(ultralytics.__version__),
        "source_dataset_yaml_sha256": file_sha256(copied_source),
        "installed_official_yaml_sha256": file_sha256(official_yaml),
        "relevant_files": {"source_yaml": str(copied_source), "training_yaml": str(dataset_yaml),
                           "installed_official_yaml": str(official_yaml),
                           "training_config": "configs/vision/carparts_yolo26n_seg_v1.toml"},
        "metrics_scope": "Official Carparts-Seg supplied TEST; TRAIN/VAL used for optimization/checkpoint selection.",
    }
    write_json(dataset_root / MANIFEST_NAME, manifest)
    write_json(manifest_path, manifest)
    return manifest


def load_carparts_dataset(dataset_yaml: Path = DEFAULT_DATASET_YAML) -> dict[str, Any]:
    """Reinspect local data and verify the frozen manifest without downloading."""
    dataset_yaml = Path(dataset_yaml).resolve()
    path = dataset_yaml.parent / MANIFEST_NAME
    if not path.is_file():
        raise DatasetValidationError(f"Prepared Carparts-Seg manifest missing: {path}; run the explicit preparation script")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    inspection = inspect_carparts_dataset(dataset_yaml)
    _require_valid(inspection)
    if manifest.get("schema_version") != 1 or manifest.get("preparation_version") != PREPARATION_VERSION:
        raise DatasetValidationError("Unsupported Carparts-Seg preparation manifest version")
    for key in ("dataset_yaml_sha256", "category_mapping", "splits", "annotation_hashes", "prepared_files", "actual_sample_count"):
        if manifest.get(key) != inspection[key]:
            raise DatasetValidationError(f"Carparts-Seg {key} changed since preparation")
    return manifest


def carparts_provenance(dataset_yaml: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    keys = ("source", "source_name", "source_repository", "dataset_name", "license", "official_splits_used",
            "preparation_version", "category_mapping", "splits", "annotation_hashes", "actual_sample_count",
            "source_dataset_yaml_sha256", "installed_official_yaml_sha256", "ultralytics_version", "metrics_scope", "source_filename_overlap_audit")
    return {key: manifest[key] for key in keys if key in manifest} | {
        "dataset_yaml_sha256": file_sha256(dataset_yaml),
        "prepared_manifest_sha256": file_sha256(Path(dataset_yaml).parent / MANIFEST_NAME),
    }
