"""Reproducible cross-validation and candidate selection."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


class CrossValidationMetrics(BaseModel):
    model_name: str
    folds: int
    f1_macro_mean: float
    f1_macro_std: float
    balanced_accuracy_mean: float
    balanced_accuracy_std: float


@dataclass
class CrossValidateConfig:
    """Model, training data and reproducible cross-validation settings."""

    name: str
    model: Pipeline
    X_train: pd.DataFrame
    y_train: pd.Series
    folds: int = 5
    seed: int = 42


def cross_validate_model(config: CrossValidateConfig) -> CrossValidationMetrics:
    cv = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    scores = cross_validate(
        config.model,
        config.X_train,
        config.y_train,
        cv=cv,
        scoring={"f1_macro": "f1_macro", "balanced_accuracy": ("balanced_accuracy")},
        n_jobs=-1,
        return_train_score=False,
    )
    return CrossValidationMetrics(
        model_name=config.name,
        folds=config.folds,
        f1_macro_mean=float(np.mean(scores["test_f1_macro"])),
        f1_macro_std=float(np.std(scores["test_f1_macro"])),
        balanced_accuracy_mean=float(np.mean(scores["test_balanced_accuracy"])),
        balanced_accuracy_std=float(np.std(scores["test_balanced_accuracy"])),
    )


def cross_validate_models(
    *,
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: int = 5,
    seed: int = 42,
) -> list[CrossValidationMetrics]:
    results = [
        cross_validate_model(
            CrossValidateConfig(name=name, model=model, X_train=X_train, y_train=y_train, folds=folds, seed=seed)
        )
        for name, model in models.items()
    ]
    return sorted(results, key=lambda item: item.f1_macro_mean, reverse=True)


def select_candidate_model(results: list[CrossValidationMetrics]) -> str:
    non_dummy = [result for result in results if result.model_name != "dummy"]
    if not non_dummy:
        raise ValueError("No non-dummy model available")
    return max(non_dummy, key=lambda result: result.f1_macro_mean).model_name
