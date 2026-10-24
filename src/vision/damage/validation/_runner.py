"""Two explicit VAL passes for standard AP and separate operational counts."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vision.damage.evaluation import extract_segmentation_metrics
from vision.damage.evaluation._metrics import write_per_class_csv
from vision.damage.training import DamageTrainingConfig, verify_class_names, write_json
from vision.damage.validation._operating import OperatingPointCollector, write_operating_csv
from vision.damage.validation._params import AP_CONFIDENCE_FLOOR, ValidationParams, ValidationRequest
from vision.damage.validation._request import prepare_validation


def _pass(model: Any, request: ValidationRequest, *, name: str, confidence: float, plots: bool):
    config = request.config
    result = model.val(
        data=str(config.dataset_yaml),
        split="val",
        conf=confidence,
        imgsz=request.image_size,
        batch=request.batch_size,
        device=config.device,
        workers=config.workers,
        project=str(request.output_dir),
        name=name,
        exist_ok=False,
        plots=plots,
    )
    directory = getattr(result, "save_dir", None)
    if directory is None or Path(directory).resolve() != (request.output_dir / name).resolve():
        raise RuntimeError("VAL diagnostics wrote to an unexpected directory")
    return result


def _payload(request, environment, ap_result, operating):
    metrics = extract_segmentation_metrics(
        ap_result, request.manifest["category_mapping"], request.manifest["splits"]["val"]
    )
    metrics["metric_semantics"] = (
        "AP uses predictions down to conf=0.001. Summary precision/recall use Ultralytics confidence curves "
        "at its maximum mean F1, not the application threshold. operating_point records separate class-aware "
        "box/mask TP, FP and FN at IoU=0.50 after rerunning VAL at the specified fixed confidence."
    )
    return {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **request.payload(),
        "diagnostic_config": request.config.payload(),
        "dataset": request.summary["dataset"],
        "environment": environment,
        "metrics_scope": "VAL from the unchanged prepared dataset used for training; never TEST.",
        "confusion_matrix_scope": "ap/confusion_matrix*.png: framework box matching on VAL, not mask confusion counts.",
        "ap_speed_ms_per_image": {key: float(value) for key, value in getattr(ap_result, "speed", {}).items()},
        **metrics,
        "operating_point": {"confidence_threshold": request.operating_confidence, **operating},
    }


def run_validation(
    config: DamageTrainingConfig,
    *,
    output_dir: Path | None = None,
    params: ValidationParams,
) -> dict[str, Any]:
    """Leave training/TEST artifacts intact and refuse to overwrite prior diagnostics."""
    request = prepare_validation(config, output_dir=output_dir, params=params)
    if params.model_factory is None or params.environment_provider is None:
        raise ValueError("VAL execution requires explicit model and environment providers")
    environment = params.environment_provider()
    model = params.model_factory(str(request.checkpoint))
    if getattr(model, "task", None) != "segment":
        raise ValueError("VAL diagnostics require an instance segmentation checkpoint")
    verify_class_names(getattr(model, "names", None), request.manifest["category_mapping"])
    request.output_dir.mkdir(parents=True, exist_ok=False)

    def capture_device(validator):
        device = getattr(validator, "device", None)
        environment["device_used"] = str(device) if device is not None else None

    model.add_callback("on_val_start", capture_device)
    ap_result = _pass(model, request, name="ap", confidence=AP_CONFIDENCE_FLOOR, plots=True)
    collector = OperatingPointCollector(request.manifest["category_mapping"])
    model.add_callback("on_val_batch_end", collector)
    _pass(model, request, name="operating", confidence=request.operating_confidence, plots=False)
    operating = collector.report(request.manifest["splits"]["val"]["image_count"])
    payload = _payload(request, environment, ap_result, operating)
    write_per_class_csv(request.output_dir / "per_class_metrics.csv", payload["per_class"])
    write_operating_csv(request.output_dir / "operating_point_metrics.csv", operating["per_class"])
    write_json(request.output_dir / "val_metrics.json", payload)
    return payload
