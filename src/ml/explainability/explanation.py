"""SHAP tree explanations over the transformed feature space."""

import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def transform_for_explanation(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


def compute_tree_shap(*, model: Pipeline, X: pd.DataFrame):
    classifier = model.named_steps["classifier"]
    transformed = transform_for_explanation(model, X)
    explainer = shap.TreeExplainer(classifier)
    explanation = explainer(transformed)
    return explanation, transformed, classifier
