"""SHAP-based triage model explainability."""

from ml.explainability.explanation import compute_tree_shap, transform_for_explanation
from ml.explainability.plots import save_class_beeswarm, save_local_waterfall

__all__ = [
    "compute_tree_shap",
    "save_class_beeswarm",
    "save_local_waterfall",
    "transform_for_explanation",
]
