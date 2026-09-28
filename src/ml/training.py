from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from pydantic import BaseModel, Field

TARGET_COLUMN = "workflow"


CATEGORICAL_FEATURES = [
    "claim_type",
    "collision_type",
]


NUMERICAL_FEATURES = [
    "vehicle_age",
    "repair_amount",
    "declared_damage_count",
    "additional_damage_count",
    "document_count",
    "missing_document_count",
    "document_completeness_ratio",
    "injuries_declared",
    "consistency_issue_count",
]


FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
)

class ModelMetrics(BaseModel):
    model_name: str

    accuracy: float = Field(
        ge=0,
        le=1,
    )

    balanced_accuracy: float = Field(
        ge=0,
        le=1,
    )

    f1_macro: float = Field(
        ge=0,
        le=1,
    )

    f1_weighted: float = Field(
        ge=0,
        le=1,
    )

    labels: list[str]

    confusion_matrix: list[
        list[int]
    ]

def load_triage_dataset(
    path: Path,
) -> pd.DataFrame:
    dataset = pd.read_csv(path)

    required_columns = set(
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = (
        required_columns
        - set(dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if dataset.empty:
        raise ValueError(
            "Dataset is empty"
        )

    if dataset[
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    ].isnull().any().any():
        raise ValueError(
            "Dataset contains missing values"
        )

    return dataset

def split_triage_dataset(
    dataset: pd.DataFrame,
    *,
    test_size: float = 0.20,
    seed: int = 42,
):
    X = dataset[
        FEATURE_COLUMNS
    ].copy()

    y = dataset[
        TARGET_COLUMN
    ].copy()

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

def build_preprocessor() -> ColumnTransformer:
    categorical = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    numerical = StandardScaler()

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical,
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                numerical,
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def _make_pipeline(
    classifier,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

def build_models(
    *,
    seed: int = 42,
) -> dict[str, Pipeline]:
    return {
        "dummy": _make_pipeline(
            DummyClassifier(
                strategy="most_frequent",
            )
        ),

        "logistic_regression": _make_pipeline(
            LogisticRegression(
                max_iter=2000,
                random_state=seed,
            )
        ),

        "random_forest": _make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            )
        ),

        "gradient_boosting": _make_pipeline(
            GradientBoostingClassifier(
                random_state=seed,
            )
        ),
    }

def evaluate_model(
    *,
    name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelMetrics:
    predictions = model.predict(
        X_test
    )

    labels = sorted(
        y_test.unique().tolist()
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    return ModelMetrics(
        model_name=name,

        accuracy=float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),

        balanced_accuracy=float(
            balanced_accuracy_score(
                y_test,
                predictions,
            )
        ),

        f1_macro=float(
            f1_score(
                y_test,
                predictions,
                average="macro",
            )
        ),

        f1_weighted=float(
            f1_score(
                y_test,
                predictions,
                average="weighted",
            )
        ),

        labels=labels,

        confusion_matrix=(
            matrix.tolist()
        ),
    )

def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int = 42,
) -> dict[str, Pipeline]:
    models = build_models(
        seed=seed
    )

    for model in models.values():
        model.fit(
            X_train,
            y_train,
        )

    return models