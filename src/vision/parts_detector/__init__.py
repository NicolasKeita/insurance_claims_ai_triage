"""Independent lazy Carparts-Seg adapter, sharing step29 image/geometry helpers."""

from vision.parts_detector._fake import FakeVehiclePartDetector
from vision.parts_detector._mapping import (
    PartMappingConfig,
    map_ultralytics_part_result,
    part_mapping_from_model_names,
)
from vision.parts_detector._settings import (
    DEFAULT_PART_MODEL_PATH,
    VehiclePartDetector,
    VehiclePartDetectorError,
    VehiclePartSettings,
)
from vision.parts_detector._ultralytics import UltralyticsVehiclePartDetector

__all__ = [
    "DEFAULT_PART_MODEL_PATH",
    "FakeVehiclePartDetector",
    "PartMappingConfig",
    "UltralyticsVehiclePartDetector",
    "VehiclePartDetector",
    "VehiclePartDetectorError",
    "VehiclePartSettings",
    "map_ultralytics_part_result",
    "part_mapping_from_model_names",
]