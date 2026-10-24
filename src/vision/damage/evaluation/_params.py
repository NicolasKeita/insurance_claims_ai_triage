"""Evaluation dependencies and the prepared request handed to the TEST run."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision.damage.training import DamageTrainingConfig


@dataclass
class EvaluationParams:
    """Evaluation dependencies, diagnostic quotas and checkpoint provenance."""

    model_factory: Callable[[str], Any]
    environment_provider: Callable[[], dict[str, Any]]
    dataset_loader: Callable[[Path], dict[str, Any]] | None = None
    max_error_examples: int = 8
    provenance_provider: Callable[[Path, dict[str, Any]], dict[str, Any]] | None = None
    confidence_threshold: float | None = None
    tracking_factory: Callable[..., Any] | None = None
    include_part_errors: bool = False
    ap_confidence: float | None = None


@dataclass(frozen=True)
class EvaluationRequest:
    """Everything the held-out TEST evaluation needs before any model is loaded."""

    config: DamageTrainingConfig
    params: EvaluationParams
    manifest: dict[str, Any]
    checkpoint: Path
    summary: dict[str, Any]
    threshold: float
    ap_confidence: float
