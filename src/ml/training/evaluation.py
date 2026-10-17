"""Supervised triage model metrics."""

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from ml.training.schema import ModelMetrics


def evaluate_model(*, name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
    predictions = model.predict(X_test)
    labels = sorted(y_test.unique().tolist())
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    return ModelMetrics(
        model_name=name,
        accuracy=float(accuracy_score(y_test, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, predictions)),
        f1_macro=float(f1_score(y_test, predictions, average="macro")),
        f1_weighted=float(f1_score(y_test, predictions, average="weighted")),
        labels=labels,
        confusion_matrix=(matrix.tolist()),
    )
