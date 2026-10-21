"""Files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from vision.dataset._models import DatasetValidationError


def _integer(value: Any, description: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetValidationError(f"{description} must be an integer")
    if minimum is not None and value < minimum:
        raise DatasetValidationError(f"{description} must be >= {minimum}")
    return value


def _relative_filename(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError("Image file_name must be a nonempty relative filename")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ":" in normalized or any(p in ("", ".", "..") for p in normalized.split("/")):
        raise DatasetValidationError(f"Unsafe relative filename: {value!r}")
    return path.as_posix()


def _contained_file(directory: Path, filename: str) -> Path:
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise DatasetValidationError(f"Image escapes its split directory: {filename}")
    if not path.is_file():
        raise DatasetValidationError(f"Image file is missing: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Cannot read JSON annotations: {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise DatasetValidationError(f"Annotations must contain a JSON object: {path}")
    return document


def annotation_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_image(path: Path, width: int, height: int) -> None:
    from PIL import Image, UnidentifiedImageError

    if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
        raise DatasetValidationError(f"Image format unsupported by this preparation pipeline: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != (width, height):
                raise DatasetValidationError(f"Image dimensions disagree with COCO/prepared metadata: {path}")
            if image.getexif().get(274, 1) not in (None, 1):
                raise DatasetValidationError(
                    f"EXIF orientation would change annotation coordinates: {path}; "
                    "source needs correctly oriented COCO annotations"
                )
            image.load()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise DatasetValidationError(f"Image is not decodable: {path}: {exc}") from exc


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
