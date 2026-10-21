"""Instances."""

from __future__ import annotations

from typing import Any

from vision.dataset import DatasetValidationError
from vision.dataset.cardd_hf._models import HfDamageInstance
from vision.dataset.cardd_hf.annotations._fields import _get, _kind, _label
from vision.dataset.cardd_hf.masks._geometry import _polygon
from vision.dataset.cardd_hf.masks._raster import _mask_polygons


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
            polygons, holes = _detection_geometry(label)
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


def _detection_geometry(label):
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
    return polygons, holes
