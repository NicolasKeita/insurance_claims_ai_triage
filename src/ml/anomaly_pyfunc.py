"""MLflow adapter for the complete anomaly detector."""

import json

import mlflow.pyfunc
import pandas as pd

from ml.anomaly_training import AnomalyDetector, select_anomaly_features


class AnomalyPyfuncModel(mlflow.pyfunc.PythonModel):
    """Persist preprocessing, calibration, and scoring as one MLflow model."""

    def __init__(self, detector: AnomalyDetector) -> None:
        self.detector = detector

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> pd.DataFrame:
        """Return one serializable assessment row for each claim."""
        features = select_anomaly_features(model_input)
        rows = []
        for claim in features.to_dict(orient="records"):
            assessment = self.detector.assess(claim)
            rows.append(
                {
                    "is_anomalous": assessment.is_anomalous,
                    "anomaly_score": assessment.anomaly_score,
                    "threshold": assessment.threshold,
                    "reference_percentile": assessment.reference_percentile,
                    "statistical_signals": json.dumps(assessment.statistical_signals),
                }
            )
        return pd.DataFrame(rows, index=model_input.index)
