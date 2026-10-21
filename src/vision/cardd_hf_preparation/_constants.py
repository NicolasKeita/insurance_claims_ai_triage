"""Shared configuration and fixed dataset contracts."""

from __future__ import annotations

from vision.dataset import (
    ROOT,
)

HF_PREPARATION_VERSION = "cardd_hf_yolo_segmentation_v1"

DEFAULT_HF_PREPARED_DIR = ROOT / "data" / "external_prepared" / "cardd_hf_v1"

DEFAULT_HF_MANIFEST_PATH = ROOT / "artifacts" / "vision" / "cardd_hf_dataset_manifest.json"
