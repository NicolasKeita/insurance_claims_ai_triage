"""Anomaly benchmark metric models."""

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, Field


class AnomalyKindMetrics(BaseModel):
    sample_count: int = Field(ge=0)
    synthetic_anomaly_count: int = Field(ge=0)
    flagged_count: int = Field(ge=0)
    flagged_rate: float = Field(ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)
    mean_anomaly_score: float


class AnomalyMetrics(BaseModel):
    sample_count: int = Field(ge=0)
    synthetic_anomaly_count: int = Field(ge=0)
    flagged_count: int = Field(ge=0)
    flagged_rate: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    confusion_matrix: list[list[int]]
    average_precision: float | None = Field(default=None, ge=0, le=1)
    roc_auc: float | None = Field(default=None, ge=0, le=1)
    threshold: float
    threshold_percentile: float = Field(ge=0, le=1)
    by_anomaly_kind: dict[str, AnomalyKindMetrics]


@dataclass
class AnomalyEvaluation:
    metrics: AnomalyMetrics
    errors: pd.DataFrame
