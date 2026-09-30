from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from sklearn.pipeline import Pipeline

def transform_for_explanation(
    model: Pipeline,
    X: pd.DataFrame,
) -> pd.DataFrame:
    preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    transformed = (
        preprocessor.transform(X)
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    return pd.DataFrame(
        transformed,
        columns=feature_names,
        index=X.index,
    )

def compute_tree_shap(
    *,
    model: Pipeline,
    X: pd.DataFrame,
):
    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    transformed = (
        transform_for_explanation(
            model,
            X,
        )
    )

    explainer = shap.TreeExplainer(
        classifier
    )

    explanation = explainer(
        transformed
    )

    return (
        explanation,
        transformed,
        classifier,
    )

def save_class_beeswarm(
    *,
    explanation,
    classifier,
    target_class: str,
    output_path: Path,
) -> None:
    classes = list(
        classifier.classes_
    )

    class_index = classes.index(
        target_class
    )
    class_explanation = (
        explanation[
            :,
            :,
            class_index,
        ]
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shap.plots.beeswarm(
        class_explanation,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

def save_local_waterfall(
    *,
    explanation,
    classifier,
    sample_index: int,
    target_class: str,
    output_path: Path,
) -> None:
    classes = list(
        classifier.classes_
    )

    class_index = classes.index(
        target_class
    )

    local_explanation = (
        explanation[
            sample_index,
            :,
            class_index,
        ]
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shap.plots.waterfall(
        local_explanation,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()