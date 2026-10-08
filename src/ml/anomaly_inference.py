"""Business-facing inference from the registered anomaly detector."""

import json
from dataclasses import dataclass

import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient
from pydantic import BaseModel

from ml.anomaly_registry import (
    ANOMALY_CANDIDATE_ALIAS,
    REGISTERED_ANOMALY_MODEL_NAME,
)
from ml.anomaly_training import ANOMALY_FEATURE_COLUMNS, AnomalyAssessment
from ml.features import TriageFeatures


class RegisteredAnomalyAssessment(BaseModel):
    assessment: AnomalyAssessment
    registered_model: str
    model_version: str
    model_alias: str


@dataclass
class RegisteredAnomalyPredictor:
    model: mlflow.pyfunc.PyFuncModel
    registered_model: str
    version: str
    alias: str

    def _to_dataframe(self, features: TriageFeatures) -> pd.DataFrame:
        payload = features.model_dump(mode="json")
        row = {column: payload[column] for column in ANOMALY_FEATURE_COLUMNS}
        return pd.DataFrame([row], columns=list(ANOMALY_FEATURE_COLUMNS))

    def assess(self, features: TriageFeatures) -> RegisteredAnomalyAssessment:
        result = self.model.predict(self._to_dataframe(features))
        if not isinstance(result, pd.DataFrame) or len(result) != 1:
            raise ValueError("Anomaly model must return one DataFrame row")

        row = result.iloc[0].to_dict()
        signals = row.get("statistical_signals")
        if not isinstance(signals, str):
            raise ValueError("Anomaly model must return JSON statistical_signals")
        row["statistical_signals"] = json.loads(signals)
        assessment = AnomalyAssessment.model_validate(row)
        return RegisteredAnomalyAssessment(
            assessment=assessment,
            registered_model=self.registered_model,
            model_version=self.version,
            model_alias=self.alias,
        )


def load_candidate_anomaly_predictor() -> RegisteredAnomalyPredictor:
    client = MlflowClient()
    version = client.get_model_version_by_alias(
        REGISTERED_ANOMALY_MODEL_NAME,
        ANOMALY_CANDIDATE_ALIAS,
    )
    model_uri = (
        f"models:/{REGISTERED_ANOMALY_MODEL_NAME}@{ANOMALY_CANDIDATE_ALIAS}"
    )
    model = mlflow.pyfunc.load_model(model_uri)
    return RegisteredAnomalyPredictor(
        model=model,
        registered_model=REGISTERED_ANOMALY_MODEL_NAME,
        version=str(version.version),
        alias=ANOMALY_CANDIDATE_ALIAS,
    )
