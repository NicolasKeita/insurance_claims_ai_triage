"""Inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _evaluate_model(model, config, ap_confidence, evaluation_dir, environment):
    validation_device: str | None = None

    def capture_validation_device(validator: Any) -> None:
        """
        Model.val uses a local validator and can leave model.device on CPU. The validator has selected
        its actual device before on_val_start.
        """
        nonlocal validation_device
        device = getattr(validator, "device", None)
        if device is not None:
            validation_device = str(device)

    add_callback = getattr(model, "add_callback", None)
    if callable(add_callback):
        add_callback("on_val_start", capture_validation_device)
    result = model.val(
        data=str(config.dataset_yaml),
        split="test",
        conf=ap_confidence,
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        workers=config.workers,
        project=str(config.output_dir),
        name="evaluation",
        exist_ok=False,
        plots=False,
    )
    actual_dir = getattr(result, "save_dir", None)
    if actual_dir is not None and Path(actual_dir).resolve() != evaluation_dir.resolve():
        raise RuntimeError("Evaluation wrote to an unexpected directory")
    _record_device(environment, validation_device, model)
    return result


def _record_device(environment, validation_device, model):
    environment["device_used"] = validation_device
    environment["device_used_source"] = (
        "validator.device observed at on_val_start"
        if validation_device is not None
        else "unobserved: validation callback did not provide a device"
    )
    if validation_device is None:
        checkpoint_device = getattr(model, "device", None)
        environment["loaded_checkpoint_device"] = str(checkpoint_device) if checkpoint_device is not None else None
