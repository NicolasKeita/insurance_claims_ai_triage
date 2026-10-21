"""Fingerprint."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from vision.cardd_hf_preparation._constants import HF_PREPARATION_VERSION
from vision.dataset import (
    SPLITS,
)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def hf_prepared_fingerprint(manifest: dict[str, Any], mapping: list[dict[str, Any]]) -> str:
    """Bind prepared geometry to the inspected source and frozen split manifest."""
    inventory_keys = (
        "source_sample_id",
        "image_id",
        "image",
        "image_sha256",
        "label",
        "label_sha256",
        "width",
        "height",
        "row_count",
        "class_counts",
    )
    inventory = {
        split: sorted(
            [
                {key: entry.get(key) for key in inventory_keys}
                for entry in manifest.get("prepared_files", {}).get(split, [])
            ],
            key=lambda entry: entry["source_sample_id"],
        )
        for split in SPLITS
    }
    return _canonical_hash(
        {
            "preparation_version": HF_PREPARATION_VERSION,
            "source_repository": manifest.get("source_repository"),
            "source_revision": manifest.get("source_revision"),
            "source_dataset_fingerprint": manifest.get("source_dataset_fingerprint"),
            "split_manifest_sha256": manifest.get("split_manifest_sha256"),
            "category_mapping": mapping,
            "annotation_hashes": manifest.get("annotation_hashes"),
            "prepared_inventory": inventory,
        }
    )
