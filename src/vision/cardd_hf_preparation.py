"""Convert inspected FiftyOne mirror instances into validated YOLO segmentation.

This module does not import FiftyOne, contact the Hub, or load a model. Source
polygons remain instance polygons; multipart instances use the existing explicit
YOLO bridge policy. Images are copied once into an atomic prepared directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vision.dataset import (
    DatasetValidationError, PREPARED_MANIFEST_NAME, ROOT, SPLITS,
    _check_image, _polygon, _try_ultralytics_conversion, _write_json,
    annotation_hash, load_prepared_dataset,
)


HF_PREPARATION_VERSION = "cardd_hf_yolo_segmentation_v1"
DEFAULT_HF_PREPARED_DIR = ROOT / "data" / "external_prepared" / "cardd_hf_v1"
DEFAULT_HF_MANIFEST_PATH = ROOT / "artifacts" / "vision" / "cardd_hf_dataset_manifest.json"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def hf_prepared_fingerprint(manifest: dict[str, Any], mapping: list[dict[str, Any]]) -> str:
    """Bind prepared geometry to the inspected source and frozen split manifest."""
    inventory_keys = ("source_sample_id", "image_id", "image", "image_sha256", "label", "label_sha256", "width", "height", "row_count", "class_counts")
    inventory = {
        split: sorted(
            [{key: entry.get(key) for key in inventory_keys} for entry in manifest.get("prepared_files", {}).get(split, [])],
            key=lambda entry: entry["source_sample_id"],
        )
        for split in SPLITS
    }
    return _canonical_hash({
        "preparation_version": HF_PREPARATION_VERSION,
        "source_repository": manifest.get("source_repository"),
        "source_revision": manifest.get("source_revision"),
        "source_dataset_fingerprint": manifest.get("source_dataset_fingerprint"),
        "split_manifest_sha256": manifest.get("split_manifest_sha256"),
        "category_mapping": mapping,
        "annotation_hashes": manifest.get("annotation_hashes"),
        "prepared_inventory": inventory,
    })


def validate_hf_prepared_manifest(manifest: dict[str, Any]) -> None:
    """Validate mirror provenance without loading original data or FiftyOne."""
    split_document = manifest.get("split_manifest")
    if not isinstance(split_document, dict) or _canonical_hash(split_document) != manifest.get("split_manifest_sha256"):
        raise DatasetValidationError("Prepared HF frozen split manifest hash disagrees")
    if manifest.get("source_name") != "CarDD HF mirror" or manifest.get("official_splits_used") is not False:
        raise DatasetValidationError("Prepared HF dataset must explicitly identify the mirror and custom splits")
    for source_key, split_key in (("source_repository", "source_repo"), ("source_revision", "source_revision"), ("actual_sample_count", "sample_count"), ("source_dataset_fingerprint", "dataset_fingerprint"), ("split_version", "split_version")):
        if manifest.get(source_key) != split_document.get(split_key):
            raise DatasetValidationError(f"Prepared HF provenance disagrees with frozen split: {source_key}")
    if not isinstance(manifest.get("source_repository"), str) or not manifest["source_repository"].strip():
        raise DatasetValidationError("Prepared HF source repository is missing")
    fields = manifest.get("annotation_fields")
    if not isinstance(fields, list) or not fields or any(not isinstance(field, str) or not field for field in fields):
        raise DatasetValidationError("Prepared HF annotation fields are missing")
    if fields != split_document.get("annotation_fields") or manifest.get("field_schema") != split_document.get("field_schema") or manifest.get("source_format") != split_document.get("source_format"):
        raise DatasetValidationError("Prepared HF annotation fields/schema disagree with frozen source")
    if manifest.get("segmentation_usable") is not True:
        raise DatasetValidationError("Prepared HF dataset requires usable instance segmentation")
    mapping = manifest.get("category_mapping", [])
    if manifest.get("actual_damage_classes") != [item.get("name") for item in mapping]:
        raise DatasetValidationError("Prepared HF damage classes disagree with category mapping")
    if split_document.get("category_mapping") != [{key: item.get(key) for key in ("category_id", "name", "yolo_class_id")} for item in mapping]:
        raise DatasetValidationError("Prepared HF category mapping disagrees with frozen split")
    sample_ids = split_document.get("sample_ids")
    if not isinstance(sample_ids, dict) or set(sample_ids) != set(SPLITS):
        raise DatasetValidationError("Prepared HF frozen split must contain train/val/test sample IDs")
    source_samples = split_document.get("samples")
    if not isinstance(source_samples, list) or any(not isinstance(row, dict) or not isinstance(row.get("sample_id"), str) for row in source_samples):
        raise DatasetValidationError("Prepared HF frozen split source sample records are invalid")
    source_lookup = {row["sample_id"]: row for row in source_samples}
    if len(source_lookup) != len(source_samples) or len(source_samples) != manifest.get("actual_sample_count"):
        raise DatasetValidationError("Prepared HF frozen source sample IDs/counts are invalid")
    label_ids = {item["name"]: str(item["yolo_class_id"]) for item in mapping}
    all_ids: set[str] = set()
    hash_splits: dict[str, str] = {}
    count = 0
    prepared_files = manifest.get("prepared_files", {})
    for split in SPLITS:
        ids = sample_ids[split]
        if not isinstance(ids, list) or not ids or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids) or all_ids.intersection(ids):
            raise DatasetValidationError(f"Prepared HF frozen split contains invalid or overlapping sample IDs: {split}")
        entries = prepared_files.get(split)
        if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries) or len(entries) != len(ids) or {entry.get("source_sample_id") for entry in entries} != set(ids):
            raise DatasetValidationError(f"Prepared HF images disagree with frozen split membership: {split}")
        for entry in entries:
            source = source_lookup.get(entry["source_sample_id"])
            if source is None or any(entry.get(entry_key) != source.get(source_key) for entry_key, source_key in (("image_sha256", "image_sha256"), ("width", "width"), ("height", "height"), ("row_count", "annotation_count"))):
                raise DatasetValidationError("Prepared HF image/instance metadata disagrees with frozen source sample")
            per_category = source.get("annotations_per_category")
            if not isinstance(per_category, dict) or any(name not in label_ids for name in per_category) or entry.get("class_counts") != {label_ids[name]: count for name, count in per_category.items()}:
                raise DatasetValidationError("Prepared HF class counts disagree with frozen source sample")
            digest = entry.get("image_sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise DatasetValidationError("Prepared HF images require SHA-256 content hashes")
            if digest in hash_splits and hash_splits[digest] != split:
                raise DatasetValidationError("Prepared HF split leakage: identical image bytes in multiple splits")
            hash_splits[digest] = split
        counts = manifest.get("splits", {}).get(split)
        frozen_counts = split_document.get("splits", {}).get(split)
        if not isinstance(counts, dict) or not isinstance(frozen_counts, dict) or any(counts.get(key) != value for key, value in frozen_counts.items()):
            raise DatasetValidationError(f"Prepared HF split distribution disagrees with frozen source: {split}")
        if counts.get("skipped_annotations") != {} or counts.get("converted_annotation_count") != frozen_counts.get("annotation_count") or counts.get("converted_annotations_per_category") != frozen_counts.get("annotations_per_category") or counts.get("converted_images_per_category") != frozen_counts.get("images_per_category"):
            raise DatasetValidationError(f"Prepared HF conversion lost instances or changed frozen distributions: {split}")
        all_ids.update(ids)
        count += len(ids)
    if manifest.get("actual_sample_count") != count or all_ids != set(source_lookup):
        raise DatasetValidationError("Prepared HF actual sample count disagrees with frozen split")


def prepare_cardd_hf_yolo(
    dataset: Any,
    output_dir: Path | str | None = None,
    *,
    split_manifest_path: Path | str | None = None,
    manifest_path: Path | str | None = DEFAULT_HF_MANIFEST_PATH,
    converter: str = "auto",
) -> dict[str, Any]:
    """Prepare every inspected instance using a persisted, immutable custom split.

    Annotation failures abort; this mirror workflow never silently drops damage
    instances. Empty images, when present in the inspected source, retain empty
    labels. Reusing a split version validates the source and does not regenerate it.
    """
    from vision.cardd_hf_split import DEFAULT_SPLIT_MANIFEST_PATH, create_or_load_cardd_hf_split
    from vision.taxonomy import map_category_name

    if converter not in ("auto", "native", "ultralytics"):
        raise DatasetValidationError("Invalid segmentation converter")
    configured = output_dir if output_dir is not None else os.environ.get("CARDD_HF_PREPARED_DIR", DEFAULT_HF_PREPARED_DIR)
    output = Path(configured).expanduser().resolve()
    source_paths = [Path(sample.filepath).resolve() for sample in dataset.samples]
    if not source_paths:
        raise DatasetValidationError("Cannot prepare an empty HF mirror")
    if any(source == output or source.is_relative_to(output) or output.is_relative_to(source.parent) for source in source_paths):
        raise DatasetValidationError("Prepared output must be outside and separate from the HF source dataset")
    if manifest_path is not None and any(Path(manifest_path).resolve().is_relative_to(source.parent) for source in source_paths):
        raise DatasetValidationError("Generated manifest must be outside the HF source dataset")
    if output.exists():
        raise DatasetValidationError(f"Prepared output already exists: {output}. Choose a new versioned --output-dir; existing data is never overwritten.")
    split_document = create_or_load_cardd_hf_split(dataset, manifest_path=split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH)
    classes = sorted(dataset.classes)
    category_mapping = [{"category_id": index + 1, "name": name, "yolo_class_id": index} for index, name in enumerate(classes)]
    try:
        enriched_mapping = [dict(item, damage_type=map_category_name(item["name"]).value) for item in category_mapping]
    except ValueError as exc:
        raise DatasetValidationError(f"Unsupported domain taxonomy category: {exc}") from exc
    if len({item["damage_type"] for item in enriched_mapping}) != len(enriched_mapping):
        raise DatasetValidationError("HF classes collapse to duplicate domain damage types")
    class_ids = {item["name"]: item["yolo_class_id"] for item in category_mapping}
    sample_lookup = {sample.sample_id: sample for sample in dataset.samples}
    image_ids = {sample_id: index for index, sample_id in enumerate(sorted(sample_lookup), start=1)}
    manifest: dict[str, Any] = {
        "preparation_version": HF_PREPARATION_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_name": "CarDD HF mirror", "source_repository": dataset.repository,
        "source_revision": dataset.revision, "source_format": dataset.source_format,
        "actual_sample_count": len(dataset.samples), "actual_damage_classes": classes,
        "field_schema": dict(dataset.field_schema), "annotation_fields": list(dataset.annotation_fields),
        "annotation_field_summaries": dict(getattr(dataset, "annotation_field_summaries", {})),
        "segmentation_usable": True, "official_splits_used": False,
        "split_version": split_document["split_version"], "split_manifest": split_document,
        "split_manifest_sha256": _canonical_hash(split_document),
        "source_dataset_fingerprint": split_document["dataset_fingerprint"],
        "category_mapping": enriched_mapping, "annotation_hashes": {}, "splits": {}, "prepared_files": {},
        "license_provenance": {"repository_file": "CarDD_license.pdf", "conditions": "The Hugging Face mirror does not supersede the original image licensing conditions. Images are not redistributed in the Git repository."},
        "metrics_scope": f"Metrics are evaluated on the project's deterministic held-out split of the {len(dataset.samples):,}-sample Hugging Face CarDD mirror, not on the official CarDD test split.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cardd-hf-preparation-", dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        documents: dict[str, dict[str, Any]] = {}
        expected: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            annotations, images, entries = [], [], []
            expected[split] = {}
            annotation_counts: Counter[str] = Counter()
            image_counts: Counter[str] = Counter()
            stitched_count = 0
            hole_count = 0
            for sample_id in split_document["sample_ids"][split]:
                sample = sample_lookup[sample_id]
                _check_image(Path(sample.filepath), sample.width, sample.height)
                if annotation_hash(Path(sample.filepath)) != sample.image_sha256:
                    raise DatasetValidationError(f"Source image changed since HF inspection: {sample_id}")
                filename = hashlib.sha256(sample_id.encode()).hexdigest() + Path(sample.filepath).suffix.lower()
                image_id = image_ids[sample_id]
                image = {"id": image_id, "file_name": filename, "width": sample.width, "height": sample.height}
                images.append(image)
                rows = []
                for instance_index, instance in enumerate(sample.instances):
                    if instance.label not in class_ids:
                        raise DatasetValidationError(f"HF instance label absent from inspected classes: {instance.label}")
                    parts = [[coordinate * (sample.width if index % 2 == 0 else sample.height) for index, coordinate in enumerate([value for point in component for value in point])] for component in instance.polygons]
                    polygon, reason = _polygon({"segmentation": parts}, sample.width, sample.height)
                    if reason or polygon is None:
                        raise DatasetValidationError(f"HF sample {sample_id} instance {instance_index}: unusable segmentation ({reason}); no bounding-box fallback")
                    class_id = class_ids[instance.label]
                    if (class_id, polygon) in rows:
                        raise DatasetValidationError(f"HF sample {sample_id}: duplicate polygon would be silently removed by YOLO")
                    rows.append((class_id, polygon))
                    stitched_count += len(parts) > 1
                    hole_count += getattr(instance, "mask_hole_count", 0)
                    annotation_counts[instance.label] += 1
                    xs, ys = polygon[::2], polygon[1::2]
                    annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": class_id + 1, "iscrowd": 0, "segmentation": [[value * (sample.width if index % 2 == 0 else sample.height) for index, value in enumerate(polygon)]], "bbox": [min(xs) * sample.width, min(ys) * sample.height, (max(xs) - min(xs)) * sample.width, (max(ys) - min(ys)) * sample.height]})
                image_counts.update({instance.label for instance in sample.instances})
                image_relative, label_relative = f"images/{split}/{filename}", f"labels/{split}/{Path(filename).stem}.txt"
                image_path, label_path = stage / image_relative, stage / label_relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sample.filepath, image_path)
                label_path.write_text("".join(str(class_id) + " " + " ".join(format(value, ".12g") for value in polygon) + "\n" for class_id, polygon in rows), encoding="utf-8")
                expected[split][filename] = rows
                entries.append({"image": image_relative, "label": label_relative, "image_id": image_id, "source_sample_id": sample_id, "image_sha256": sample.image_sha256, "width": sample.width, "height": sample.height, "row_count": len(rows), "class_counts": dict(Counter(str(row[0]) for row in rows))})
            documents[split] = {"images": images, "annotations": annotations, "categories": [{"id": item["category_id"], "name": item["name"]} for item in category_mapping]}
            manifest["annotation_hashes"][split] = _canonical_hash(documents[split])
            manifest["prepared_files"][split] = entries
            per_category = {name: annotation_counts[name] for name in classes}
            per_class_images = {name: image_counts[name] for name in classes}
            unlabeled = sum(entry["row_count"] == 0 for entry in entries)
            manifest["splits"][split] = {"image_count": len(entries), "annotation_count": len(annotations), "annotations_per_category": per_category, "images_per_category": per_class_images, "images_without_annotations": unlabeled, "converted_annotation_count": len(annotations), "skipped_annotations": {}, "skipped_annotations_per_category": {}, "stitched_multipart_annotations": stitched_count, "source_mask_holes": hole_count, "label_count": len(entries), "images_without_converted_annotations": unlabeled, "converted_annotations_per_category": per_category, "converted_images_per_category": per_class_images}
        conversion = {"backend": "native_polygon", "unsupported_policy": "error", "segmentation_enabled": True, "bbox_fallback": False, "source_format": "FiftyOne normalized instance polygons", "multipart_policy": "preserve component vertices using retraced nearest-vertex bridges", "multipart_limitation": "YOLO uses one polygon per instance; disconnected topology and holes cannot be represented exactly, and rasterization may include bridge pixels", "use_segments": True, "cls91to80": False}
        if converter != "native":
            try:
                _try_ultralytics_conversion(stage, documents, expected, len(classes))
                conversion["backend"] = "ultralytics.convert_coco"
            except Exception as exc:
                if converter == "ultralytics":
                    raise DatasetValidationError(f"COCO-format converter failed HF polygon validation: {exc}") from exc
                conversion["fallback_reason"] = f"{type(exc).__name__}: converter unavailable, incompatible, or failed strict polygon validation"
        manifest["conversion"] = conversion
        for entries in manifest["prepared_files"].values():
            for entry in entries:
                entry["label_sha256"] = annotation_hash(stage / entry["label"])
        manifest["dataset_fingerprint"] = hf_prepared_fingerprint(manifest, category_mapping)
        dataset_yaml = {"path": str(stage), **{name: f"images/{name}" for name in SPLITS}, "names": classes, "nc": len(classes)}
        _write_json(stage / "dataset.yaml", dataset_yaml)
        _write_json(stage / PREPARED_MANIFEST_NAME, manifest)
        load_prepared_dataset(stage / "dataset.yaml")
        for scratch in (stage / "converter_annotations", stage / "converter_output"):
            if scratch.exists():
                shutil.rmtree(scratch)
        dataset_yaml["path"] = str(output)
        _write_json(stage / "dataset.yaml", dataset_yaml)
        stage.rename(output)
    if manifest_path is not None:
        _write_json(Path(manifest_path), manifest)
    return manifest
