"""Persisted SHAP beeswarm and waterfall figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import shap


def save_class_beeswarm(*, explanation, classifier, target_class: str, output_path: Path) -> None:
    classes = list(classifier.classes_)
    class_index = classes.index(target_class)
    class_explanation = explanation[:, :, class_index]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.plots.beeswarm(class_explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_local_waterfall(*, explanation, classifier, sample_index: int, target_class: str, output_path: Path) -> None:
    classes = list(classifier.classes_)
    class_index = classes.index(target_class)
    local_explanation = explanation[sample_index, :, class_index]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.plots.waterfall(local_explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
