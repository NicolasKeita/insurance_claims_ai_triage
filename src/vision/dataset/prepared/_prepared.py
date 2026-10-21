"""Prepared."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vision.dataset._constants import (
    PREPARATION_VERSION,
    PREPARED_MANIFEST_NAME,
    SPLITS,
)
from vision.dataset._files import _read_json
from vision.dataset._models import DatasetValidationError
from vision.dataset.coco._discovery import read_category_mapping
from vision.dataset.prepared._inventory import _validate_prepared_splits


def load_prepared_dataset(dataset_yaml: Path) -> dict[str, Any]:
    """Verify dataset/manifest, actual images, labels, counts and split isolation."""
    document, dataset_root, manifest, is_hf_mirror = _prepared_header(dataset_yaml)
    mapping = _prepared_provenance(document, manifest, is_hf_mirror)
    _validate_prepared_splits(document, dataset_root, manifest, is_hf_mirror, mapping)
    return manifest


def _prepared_header(dataset_yaml):
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
    from vision.dataset.cardd_hf_preparation import HF_PREPARATION_VERSION

    is_hf_mirror = manifest.get("preparation_version") == HF_PREPARATION_VERSION
    if manifest.get("preparation_version") != PREPARATION_VERSION and not is_hf_mirror:
        raise DatasetValidationError("Unsupported prepared dataset manifest version")
    if is_hf_mirror:
        from vision.dataset.cardd_hf_preparation import validate_hf_prepared_manifest

        validate_hf_prepared_manifest(manifest)
    return document, dataset_root, manifest, is_hf_mirror


def _prepared_provenance(document, manifest, is_hf_mirror):
    mapping, valid_mapping = _prepared_mapping(manifest)
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
        from vision.dataset.cardd_hf_preparation import hf_prepared_fingerprint

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
    return mapping


def _prepared_mapping(manifest):
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
    return mapping, valid_mapping
