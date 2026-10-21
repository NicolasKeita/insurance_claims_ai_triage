"""Request."""

from __future__ import annotations

import math

from vision.damage.evaluation._checkpoint import selected_checkpoint
from vision.damage.evaluation._params import EvaluationRequest


def _prepare_request(config, params):
    max_error_examples = params.max_error_examples
    dataset_loader, provenance_provider = params.dataset_loader, params.provenance_provider
    confidence_threshold = params.confidence_threshold
    if max_error_examples < 0:
        raise ValueError("max_error_examples must be non-negative")
    if dataset_loader is None:
        from vision.dataset import load_prepared_dataset

        dataset_loader = load_prepared_dataset
    manifest = dataset_loader(config.dataset_yaml)
    checkpoint, summary = (
        selected_checkpoint(config, manifest)
        if provenance_provider is None
        else selected_checkpoint(config, manifest, provenance_provider)
    )
    from vision.config import VisionSettings

    threshold = VisionSettings.from_env().confidence_threshold if confidence_threshold is None else confidence_threshold
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be finite and between zero and one")
    if threshold != summary.get("confidence_threshold"):
        raise ValueError(
            "Confidence threshold differs from training metadata. "
            "Configure it before training or calibrate on VAL in a new version; never select it on TEST."
        )
    evaluation_dir = config.output_dir / "evaluation"
    metrics_path = config.output_dir / "test_metrics.json"
    if evaluation_dir.exists() or metrics_path.exists():
        raise FileExistsError("Held-out evaluation artifacts already exist; choose a new experiment version")
    return EvaluationRequest(config, params, manifest, checkpoint, summary, threshold)
