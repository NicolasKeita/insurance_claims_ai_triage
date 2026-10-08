from dataclasses import dataclass

import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from claims.enums import TriageWorkflow
from ml.features import TriageFeatures
from ml.registry import CANDIDATE_ALIAS, REGISTERED_MODEL_NAME
from ml.training import FEATURE_COLUMNS


class TriagePrediction(BaseModel):
    recommended_workflow: TriageWorkflow
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    registered_model: str
    model_version: str
    model_alias: str


@dataclass
class RegisteredTriagePredictor:
    model: Pipeline
    registered_model: str
    version: str
    alias: str

    def _to_dataframe(self, features: TriageFeatures) -> pd.DataFrame:
        payload = features.model_dump(mode="json")
        row = {column: payload[column] for column in FEATURE_COLUMNS}
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def predict(self, features: TriageFeatures) -> TriagePrediction:
        frame = self._to_dataframe(features)
        prediction = self.model.predict(frame)[0]
        probability_values = self.model.predict_proba(frame)[0]
        classes = self.model.named_steps["classifier"].classes_
        probabilities = {
            str(label): float(probability)
            for label, probability in zip(classes, probability_values, strict=True)
        }
        workflow = TriageWorkflow(str(prediction))
        return TriagePrediction(
            recommended_workflow=workflow,
            confidence=probabilities[workflow.value],
            probabilities=probabilities,
            registered_model=self.registered_model,
            model_version=self.version,
            model_alias=self.alias,
        )


def load_candidate_predictor() -> RegisteredTriagePredictor:
    client = MlflowClient()
    version = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, CANDIDATE_ALIAS
    )
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{CANDIDATE_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)
    return RegisteredTriagePredictor(
        model=model,
        registered_model=REGISTERED_MODEL_NAME,
        version=version.version,
        alias=CANDIDATE_ALIAS,
    )
