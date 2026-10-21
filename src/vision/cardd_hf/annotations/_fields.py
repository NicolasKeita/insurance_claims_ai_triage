"""Fields."""

from __future__ import annotations

import re
from typing import Any

from vision.dataset import DatasetValidationError
from vision.taxonomy import map_category_name


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    try:
        return value[key]
    except (TypeError, KeyError, AttributeError, ValueError):
        return getattr(value, key, default)


def _kind(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("_cls", "")).rsplit(".", 1)[-1]
    return type(value).__name__


def _schema_kind(value: Any) -> str:
    document_type = getattr(value, "document_type", None)
    if document_type is not None:
        return document_type.__name__
    text = str(value)
    match = re.search(r"\b(Detections|Detection|Polylines|Polyline|Segmentation)\b", text)
    return match.group(1) if match else ""


def _label(value: Any) -> str:
    name = _get(value, "label")
    if not isinstance(name, str) or not name.strip():
        raise DatasetValidationError("Instance annotation has no nonempty class label")
    try:
        map_category_name(name)
    except ValueError as exc:
        raise DatasetValidationError(str(exc)) from exc
    return name.strip()
