"""Lazy Ultralytics adapter and a deterministic fake for dataset-free tests."""

from vision.damage.detector._checkpoint import local_model_metadata, require_local_checkpoint
from vision.damage.detector._errors import DamageDetector, DamageDetectorError, DetectorOutputError
from vision.damage.detector._fake import FakeDamageDetector
from vision.damage.detector._mapping import (
    DamageMappingConfig,
    category_mapping_from_model_names,
    map_ultralytics_result,
)
from vision.damage.detector._output import _list, extract_ultralytics_rows, normalized_detection_geometry
from vision.damage.detector._ultralytics import UltralyticsDamageDetector

__all__ = [
    "DamageDetector",
    "DamageDetectorError",
    "DamageMappingConfig",
    "DetectorOutputError",
    "FakeDamageDetector",
    "UltralyticsDamageDetector",
    "_list",
    "category_mapping_from_model_names",
    "extract_ultralytics_rows",
    "local_model_metadata",
    "map_ultralytics_result",
    "normalized_detection_geometry",
    "require_local_checkpoint",
]
