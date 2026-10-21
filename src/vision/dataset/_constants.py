"""Shared configuration and fixed dataset contracts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

SPLITS = ("train", "val", "test")

PREPARATION_VERSION = "cardd_yolo_segmentation_v1"

DEFAULT_PREPARED_DIR = ROOT / "data" / "external_prepared" / "cardd_yolo"

DEFAULT_MANIFEST_PATH = ROOT / "artifacts" / "vision" / "cardd_dataset_manifest.json"

PREPARED_MANIFEST_NAME = "cardd_dataset_manifest.json"
