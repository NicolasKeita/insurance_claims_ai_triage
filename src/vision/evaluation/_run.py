"""Held-out TEST evaluation with an observed validator device and artifact checks."""

import math
from typing import Any

from vision.config import VisionSettings
from vision.evaluation._artifacts import _add_error_analysis, _track, _write_base_artifacts
from vision.evaluation._checkpoint import selected_checkpoint
from vision.evaluation._params import EVALUATION_DIR_NAME, EvaluationParams, EvaluationRequest
from vision.evaluation._validator import _evaluate


def run_evaluation(config, *, params: EvaluationParams) -> dict[str, Any]:
    """
    Validator save_dir is observed rather than assumed, to catch auto-increment.
    """
    _validate_params(params)
    request = _prepare_request(config, params)
    payload, model = _evaluate(request)
    metrics_path, csv_path = _write_base_artifacts(payload, request)
    _add_error_analysis(payload, model, request, params, metrics_path)
    _track(payload, request, params, (metrics_path, csv_path))
    return payload


def _validate_params(params):
    if params.max_error_examples < 0:
        raise ValueError("max_error_examples must be non-negative")


def _prepare_request(config, params) -> EvaluationRequest:
    dataset_loader = params.dataset_loader
    if dataset_loader is None:
        from vision.dataset import load_prepared_dataset

        dataset_loader = load_prepared_dataset
    manifest = dataset_loader(config.dataset_yaml)
    checkpoint, summary = (
        selected_checkpoint(config, manifest)
        if params.provenance_provider is None
        else selected_checkpoint(config, manifest, params.provenance_provider)
    )
    threshold = (
        VisionSettings.from_env().confidence_threshold
        if params.confidence_threshold is None
        else params.confidence_threshold
    )
    _validate_threshold(threshold, summary)
    evaluation_dir = config.output_dir / EVALUATION_DIR_NAME
    if evaluation_dir.exists() or (config.output_dir / "test_metrics.json").exists():
        raise FileExistsError("Held-out evaluation artifacts already exist; choose a new experiment version")
    return EvaluationRequest(
        config=config,
        params=params,
        manifest=manifest,
        checkpoint=checkpoint,
        summary=summary,
        threshold=threshold,
    )


def _validate_threshold(threshold, summary):
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be finite and between zero and one")
    if threshold != summary.get("confidence_threshold"):
        raise ValueError(
            "Confidence threshold differs from training metadata. "
            "Configure it before training or calibrate on VAL in a new version; never select it on TEST."
        )
