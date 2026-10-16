"""Final official TEST evaluation via the shared segmentation evaluator."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from vision.carparts_dataset import carparts_provenance, load_carparts_dataset
from vision.evaluation import run_evaluation
from vision.parts_training import VehiclePartTrainingConfig, require_requested_device


def run_parts_evaluation(
    config: VehiclePartTrainingConfig,
    *,
    model_factory: Callable[[str], Any],
    environment_provider: Callable[[], dict[str, Any]],
    max_error_examples: int = 8,
    dataset_loader: Callable[[Path], dict[str, Any]] = load_carparts_dataset,
    tracking_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    from vision.parts_tracking import parts_tracking_run

    return run_evaluation(
        config,
        model_factory=model_factory,
        environment_provider=lambda: require_requested_device(config, environment_provider()),
        dataset_loader=dataset_loader,
        max_error_examples=max_error_examples,
        provenance_provider=carparts_provenance,
        confidence_threshold=config.confidence_threshold,
        tracking_factory=tracking_factory or parts_tracking_run,
        include_part_errors=True,
    )
