"""Bounded diagnostics; box matching and polygon mask overlap are not mask AP."""

from vision.error_analysis._examples import save_error_examples
from vision.error_analysis._matching import analyze_instances
from vision.error_analysis._models import (
    AnalysisConfig,
    ErrorExamplesConfig,
    ErrorInstance,
)
from vision.error_analysis._overlap import (
    box_iou,
    polygon_mask_iou,
    polygon_mask_overlap,
)

__all__ = [
    "AnalysisConfig",
    "ErrorExamplesConfig",
    "ErrorInstance",
    "analyze_instances",
    "box_iou",
    "polygon_mask_iou",
    "polygon_mask_overlap",
    "save_error_examples",
]
