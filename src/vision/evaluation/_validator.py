"""Model loading, validator run and TEST payload assembly."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vision.evaluation._metrics import extract_segmentation_metrics
from vision.evaluation._params import EVALUATION_DIR_NAME
from vision.training import verify_class_names


def _evaluate(request) -> tuple[dict[str, Any], Any]:
    environment = request.params.environment_provider()
    model = request.params.model_factory(str(request.checkpoint))
    if getattr(model, "task", None) != "segment":
        raise ValueError("Selected checkpoint must have task=segment")
    verify_class_names(getattr(model, "names", None), request.manifest["category_mapping"])
    result, device = _run_validator(model, request)
    _record_device(environment, model, device)
    return _payload(request, result, environment), model


def _record_device(environment, model, device):
    environment["device_used"] = device
    environment["device_used_source"] = (
        "validator.device observed at on_val_start"
        if device is not None
        else "unobserved: validation callback did not provide a device"
    )
    if device is None:
        checkpoint_device = getattr(model, "device", None)
        environment["loaded_checkpoint_device"] = str(checkpoint_device) if checkpoint_device is not None else None


def _run_validator(model, request):
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
        data=str(request.config.dataset_yaml),
        split="test",
        conf=request.threshold,
        imgsz=request.config.imgsz,
        batch=request.config.batch,
        device=request.config.device,
        workers=request.config.workers,
        project=str(request.config.output_dir),
        name=EVALUATION_DIR_NAME,
        exist_ok=False,
        plots=False,
    )
    actual_dir = getattr(result, "save_dir", None)
    expected_dir = (request.config.output_dir / EVALUATION_DIR_NAME).resolve()
    if actual_dir is not None and Path(actual_dir).resolve() != expected_dir:
        raise RuntimeError("Evaluation wrote to an unexpected directory")
    return result, validation_device


def _payload(request, result, environment) -> dict[str, Any]:
    summary, config = request.summary, request.config
    return {
        "schema_version": 1,
        "split": "test",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": "train/weights/best.pt",
        "checkpoint_sha256": summary["best_checkpoint_sha256"],
        "checkpoint_selection_split": "val",
        "confidence_threshold": request.threshold,
        "threshold_selection": "configured baseline; no TEST threshold tuning",
        "environment": environment,
        "evaluation_config": config.payload(),
        "dataset": summary["dataset"],
        "metrics_scope": request.manifest.get("metrics_scope", "Evaluation on the configured local COCO test split."),
        **extract_segmentation_metrics(
            result, request.manifest["category_mapping"], request.manifest["splits"]["test"]
        ),
    }
