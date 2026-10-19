"""Triage training schema, feature columns and metric model."""

from pydantic import BaseModel, Field

TARGET_COLUMN = "workflow"
CATEGORICAL_FEATURES = ["claim_type", "collision_type"]
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
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float = Field(ge=0, le=1)
    balanced_accuracy: float = Field(ge=0, le=1)
    f1_macro: float = Field(ge=0, le=1)
    f1_weighted: float = Field(ge=0, le=1)
    labels: list[str]
    confusion_matrix: list[list[int]]
