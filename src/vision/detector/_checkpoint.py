"""Checkpoint presence checks and local weight identity snapshots."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from vision.detector._errors import DamageDetectorError
from vision.models import VisionModelMetadata


def require_local_checkpoint(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_file() or path.suffix.lower() != ".pt" or path.stat().st_size == 0:
        raise DamageDetectorError(
            f"Trained damage checkpoint unavailable: {path}. Prepare CarDD and run "
            "scripts/train_damage_segmentation.py, or set DAMAGE_MODEL_PATH to an existing "
            "trained damage segmentation .pt file. No weights were downloaded."
        )
    return path.resolve()


def local_model_metadata(path: Path) -> VisionModelMetadata:
    """Snapshot identity of local weights without loading any ML framework."""
    path = Path(path).resolve()
    digest = None
    if path.is_file():
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(chunk)
        digest = f"sha256:{sha.hexdigest()}"
    try:
        library_version = version("ultralytics")
    except PackageNotFoundError:
        library_version = None
    return VisionModelMetadata(
        identifier=str(path),
        version=digest,
        framework="ultralytics",
        framework_version=library_version,
    )