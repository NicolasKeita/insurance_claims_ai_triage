"""Validate provenance and destinations before any model or framework is loaded."""

import math
from pathlib import Path

from vision.config import ROOT
from vision.damage.evaluation import selected_checkpoint
from vision.damage.training import DamageTrainingConfig
from vision.damage.validation._params import ValidationParams, ValidationRequest


def _operating_confidence(params: ValidationParams) -> float:
    value = params.operating_confidence
    if value is None:
        from vision.config import VisionSettings

        value = VisionSettings.from_env().confidence_threshold
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("Operating confidence must be finite and between zero and one")
    return float(value)


def _destination(config: DamageTrainingConfig, image_size: int, output_dir: Path | None) -> Path:
    path = output_dir or ROOT / "artifacts/vision/damage_diagnostics" / config.version / f"val_imgsz_{image_size}"
    path = Path(path).expanduser()
    path = ((ROOT / path) if not path.is_absolute() else path).resolve()
    if path.is_relative_to(config.output_dir.resolve()):
        raise ValueError("VAL diagnostics must use a separate directory outside the training run")
    if path.is_relative_to(config.dataset_yaml.parent.resolve()):
        raise ValueError("VAL diagnostics must not write inside the prepared dataset")
    if path.exists():
        raise FileExistsError(f"VAL diagnostic output already exists; choose a new --output-dir: {path}")
    return path


def prepare_validation(
    config: DamageTrainingConfig,
    *,
    output_dir: Path | None = None,
    params: ValidationParams | None = None,
) -> ValidationRequest:
    """Read existing dataset/checkpoint metadata only; do not create output directories."""
    params = params or ValidationParams()
    image_size = config.imgsz if params.image_size is None else params.image_size
    if type(image_size) is not int or image_size <= 0:
        raise ValueError("Diagnostic image size must be a positive integer")
    batch_size = config.batch if params.batch_size is None else params.batch_size
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Diagnostic batch size must be a positive integer")
    threshold = _operating_confidence(params)
    destination = _destination(config, image_size, output_dir)
    loader = params.dataset_loader
    if loader is None:
        from vision.dataset import load_prepared_dataset

        loader = load_prepared_dataset
    manifest = loader(config.dataset_yaml)
    if "val" not in manifest.get("splits", {}):
        raise ValueError("Prepared dataset has no VAL split")
    checkpoint, summary = selected_checkpoint(config, manifest, params.provenance_provider)
    return ValidationRequest(
        config, params, destination, manifest, checkpoint, summary, threshold, image_size, batch_size
    )
