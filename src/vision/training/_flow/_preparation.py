"""Validate the prepared dataset and the recovery baseline before any import."""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vision.config import VisionSettings
from vision.training._artifacts import dataset_provenance
from vision.training._config import DamageTrainingConfig, TrainingHooks
from vision.training._records import PreparedRun
from vision.training._resume import load_resume_metadata


def _dataset_loader(hooks: TrainingHooks) -> Callable[[Path], dict[str, Any]]:
    if hooks.dataset_loader is not None:
        return hooks.dataset_loader
    from vision.dataset import load_prepared_dataset

    return load_prepared_dataset


def _confidence_threshold(hooks: TrainingHooks) -> float:
    if hooks.confidence_threshold is None:
        threshold = VisionSettings.from_env().confidence_threshold
    else:
        threshold = hooks.confidence_threshold
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be finite and between zero and one")
    return threshold


def prepare_run(config: DamageTrainingConfig, hooks: TrainingHooks) -> PreparedRun:
    """Validate the prepared dataset and the recovery baseline before any framework import."""
    manifest = _dataset_loader(hooks)(config.dataset_yaml)
    threshold = _confidence_threshold(hooks)
    provenance_provider = hooks.provenance_provider or dataset_provenance
    recovery = (
        load_resume_metadata(config, manifest, threshold, provenance_provider=provenance_provider)
        if hooks.resume
        else None
    )
    if config.output_dir.exists() and not hooks.resume:
        raise FileExistsError(f"Run directory already exists: {config.output_dir}. Choose a new version/output_dir.")
    return PreparedRun(
        manifest=manifest,
        threshold=threshold,
        provenance_provider=provenance_provider,
        recovery=recovery,
    )
