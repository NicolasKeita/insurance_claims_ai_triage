"""Small, explicit image validation before detector invocation."""

import warnings
from pathlib import Path

from vision.config import VisionSettings

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "TIFF", "BMP"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


class ImageValidationError(ValueError):
    pass


def load_validated_image(path: Path, settings: VisionSettings | None = None):
    """Return a detached RGB Pillow image after applying EXIF orientation.

    The caller owns the returned image and should close it after use. No manual
    resize/normalization is applied; Ultralytics handles model preprocessing.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    path = Path(path)
    settings = settings or VisionSettings()
    if not path.is_file():
        raise ImageValidationError(f"Image file does not exist: {path}")
    if path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError(f"Unsupported image extension: {path.suffix}")
    size = path.stat().st_size
    if not 0 < size <= settings.max_image_bytes:
        raise ImageValidationError(f"Image file is empty or exceeds {settings.max_image_bytes} bytes: {path}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format not in SUPPORTED_FORMATS:
                    raise ImageValidationError(f"Unsupported decoded image format: {image.format}")
                width, height = image.size
                if width <= 0 or height <= 0 or width * height > settings.max_image_pixels:
                    raise ImageValidationError(f"Image dimensions exceed the configured pixel limit: {path}")
                image.load()
                oriented = ImageOps.exif_transpose(image)
                try:
                    return oriented.convert("RGB")
                finally:
                    oriented.close()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ImageValidationError(f"Image cannot be safely decoded: {path}") from error
