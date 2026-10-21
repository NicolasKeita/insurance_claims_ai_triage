"""Dataset YAML loading and strict class-mapping validation."""

from pathlib import Path
from typing import Any

from vision.dataset import DatasetValidationError


def read_yaml(path: Path) -> dict[str, Any]:
    """
    PyYAML is a small transitive vision dependency, imported only on inspection.
    """
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise DatasetValidationError(f"Dataset YAML must contain an object: {path}")
    return document


def class_mapping(document: dict[str, Any]) -> list[dict[str, Any]]:
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
    if any(not isinstance(name, str) or not name.strip() for name in names.values()) or len(set(names.values())) != len(
        names
    ):
        raise DatasetValidationError("Dataset class names must be unique nonempty strings")
    if document.get("nc", len(names)) != len(names):
        raise DatasetValidationError("Dataset nc disagrees with class names")
    return [{"yolo_class_id": index, "category_id": index, "name": names[index]} for index in range(len(names))]
