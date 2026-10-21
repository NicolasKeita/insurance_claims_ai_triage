"""Support."""

import pandas as pd

from ml.features import TriageFeatures


def make_features() -> TriageFeatures:
    return TriageFeatures(
        claim_type="AUTO_COLLISION",
        collision_type="FRONT_COLLISION",
        vehicle_age=5,
        repair_amount=3160,
        declared_damage_count=2,
        additional_damage_count=1,
        document_count=4,
        missing_document_count=1,
        document_completeness_ratio=0.75,
        injuries_declared=False,
        consistency_issue_count=1,
    )


class FakePyfuncModel:
    def __init__(self) -> None:
        self.input_frame: pd.DataFrame | None = None

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.input_frame = frame
        return pd.DataFrame(
            [
                {
                    "is_anomalous": True,
                    "anomaly_score": 0.72,
                    "threshold": 0.61,
                    "reference_percentile": 0.995,
                    "statistical_signals": '["repair_amount: very_high"]',
                }
            ]
        )
