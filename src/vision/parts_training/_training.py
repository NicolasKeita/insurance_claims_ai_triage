"""Vehicle-part baseline using the shared segmentation orchestration."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from vision.carparts_dataset import carparts_provenance, load_carparts_dataset
from vision.parts_training._config import VehiclePartTrainingConfig, require_requested_device
from vision.parts_training._gpu_evidence import observed_model
from vision.training import TrainingHooks, build_hooks, run_training


@dataclass(frozen=True)
class PartsTrainingHooks:
    """Only the model and environment are mandatory; the rest have safe defaults."""

    model_factory: Callable[[str], Any]
    environment_provider: Callable[[], dict[str, Any]]
    resume: bool = False
    dataset_loader: Callable[[Path], dict[str, Any]] = load_carparts_dataset
    tracking_factory: Callable[..., Any] | None = None


def _training_hooks(
    config: VehiclePartTrainingConfig,
    hooks: PartsTrainingHooks,
    default_tracking: Callable[..., Any],
) -> TrainingHooks:
    return TrainingHooks(
        model_factory=partial(observed_model, config, hooks.model_factory),
        environment_provider=lambda: require_requested_device(config, hooks.environment_provider()),
        dataset_loader=hooks.dataset_loader,
        resume=hooks.resume,
        provenance_provider=carparts_provenance,
        confidence_threshold=config.confidence_threshold,
        tracking_factory=hooks.tracking_factory or default_tracking,
    )


def run_parts_training(
    config: VehiclePartTrainingConfig,
    hooks: PartsTrainingHooks | None = None,
    **hook_overrides: Any,
) -> dict[str, Any]:
    """Train the part baseline; recording stays explicit and training-only.

    Keyword hook arguments (``model_factory``, ``environment_provider``,
    ``dataset_loader``, ``resume``, ``tracking_factory``) remain supported.
    """
    from vision.parts_tracking import parts_tracking_run

    hooks = build_hooks(hooks, hook_overrides, PartsTrainingHooks)
    return run_training(config, _training_hooks(config, hooks, parts_tracking_run))
