from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.inspection import (
    permutation_importance,
)
from sklearn.metrics import (
    classification_report,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
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
    cv = StratifiedKFold(
        n_splits=config.folds,
        shuffle=True,
        random_state=config.seed,
    )
    scores = cross_validate(
        config.model,
        config.X_train,
        config.y_train,
        cv=cv,
        scoring={
            "f1_macro": "f1_macro",
            "balanced_accuracy": ("balanced_accuracy"),
        },
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
            CrossValidateConfig(
                name=name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                folds=folds,
                seed=seed,
            )
        )
        for name, model in models.items()
    ]

    return sorted(
        results,
        key=lambda item: item.f1_macro_mean,
        reverse=True,
    )


def select_candidate_model(
    results: list[CrossValidationMetrics],
) -> str:
    non_dummy = [result for result in results if result.model_name != "dummy"]

    if not non_dummy:
        raise ValueError("No non-dummy model available")

    return max(
        non_dummy,
        key=lambda result: result.f1_macro_mean,
    ).model_name


def build_classification_report(
    *,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report).transpose()


def build_error_analysis(
    *,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)
    result = X_test.copy()

    result["actual"] = y_test.to_numpy()

    result["predicted"] = predictions

    result["prediction_confidence"] = probabilities.max(axis=1)
    errors = result[result["actual"] != result["predicted"]].copy()
    return errors.sort_values(
        by="prediction_confidence",
        ascending=False,
    )


def compute_permutation_importance(
    *,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int = 42,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=20,
        random_state=seed,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": (X_test.columns),
            "importance_mean": (result.importances_mean),
            "importance_std": (result.importances_std),
        }
    )

    return importance.sort_values(
        by="importance_mean",
        ascending=False,
    )
