"""Injected collaborators of the final TEST part evaluation."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision.carparts_dataset import carparts_provenance, load_carparts_dataset
from vision.evaluation import EvaluationParams, run_evaluation
from vision.parts_training import VehiclePartTrainingConfig, require_requested_device
from vision.training import build_hooks


@dataclass(frozen=True)
class PartsEvaluationHooks:
    """Only the model and environment are mandatory; the rest have safe defaults."""

    model_factory: Callable[[str], Any]
    environment_provider: Callable[[], dict[str, Any]]
    dataset_loader: Callable[[Path], dict[str, Any]] = load_carparts_dataset
    max_error_examples: int = 8
    tracking_factory: Callable[..., Any] | None = None


def run_parts_evaluation(
    config: VehiclePartTrainingConfig,
    hooks: PartsEvaluationHooks | None = None,
    **hook_overrides: Any,
) -> dict[str, Any]:
    """Final official TEST evaluation via the shared segmentation evaluator.

    Keyword hook arguments (``model_factory``, ``environment_provider``,
    ``max_error_examples``, ...) remain supported.
    """
    from vision.parts_tracking import parts_tracking_run

    hooks = build_hooks(hooks, hook_overrides, PartsEvaluationHooks)
    return run_evaluation(
        config,
        params=EvaluationParams(
            model_factory=hooks.model_factory,
            environment_provider=lambda: require_requested_device(config, hooks.environment_provider()),
            dataset_loader=hooks.dataset_loader,
            max_error_examples=hooks.max_error_examples,
            provenance_provider=carparts_provenance,
            confidence_threshold=config.confidence_threshold,
            tracking_factory=hooks.tracking_factory or parts_tracking_run,
            include_part_errors=True,
        ),
    )
