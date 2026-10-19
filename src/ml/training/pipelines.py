"""Triage preprocessing, model candidates and fitting."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.training.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numerical = StandardScaler()
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("numerical", numerical, NUMERICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _make_pipeline(classifier) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", classifier)])


def build_models(*, seed: int = 42) -> dict[str, Pipeline]:
    return {
        "dummy": _make_pipeline(DummyClassifier(strategy="most_frequent")),
        "logistic_regression": _make_pipeline(LogisticRegression(max_iter=2000, random_state=seed)),
        "random_forest": _make_pipeline(
            RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        ),
        "gradient_boosting": _make_pipeline(GradientBoostingClassifier(random_state=seed)),
    }


def train_models(X_train: pd.DataFrame, y_train: pd.Series, *, seed: int = 42) -> dict[str, Pipeline]:
    models = build_models(seed=seed)
    for model in models.values():
        model.fit(X_train, y_train)
    return models
