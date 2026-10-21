"""Models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorInstance:
    class_id: int
    box: tuple[float, float, float, float]
    confidence: float | None = None
    polygon: tuple[tuple[float, float], ...] = ()


@dataclass
class AnalysisConfig:
    """Confidence, matching and optional mask diagnostic settings."""

    confidence_threshold: float
    matching_iou: float = 0.5
    image_size: tuple[int, int] | None = None
    include_part_errors: bool = False


@dataclass
class ErrorExamplesConfig:
    """Limits and optional vehicle-part diagnostics for error overlays."""

    max_examples: int = 8
    max_images: int = 100
    include_part_errors: bool = False
