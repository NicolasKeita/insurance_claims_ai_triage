"""Test roundtrip."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml.anomaly_tracking import (
    verify_anomaly_model_round_trip,
)


def test_round_trip_detects_score_change():
    class Detector:
        def assess(self, row):
            return SimpleNamespace(
                is_anomalous=False,
                anomaly_score=0.4,
                threshold=0.6,
                reference_percentile=0.5,
                statistical_signals=[],
            )

    class Model:
        def __init__(self, score):
            self.score = score

        def predict(self, frame):
            return pd.DataFrame(
                {
                    "is_anomalous": [False],
                    "anomaly_score": [self.score],
                    "threshold": [0.6],
                    "reference_percentile": [0.5],
                    "statistical_signals": ["[]"],
                }
            )

    from ml.anomaly_training import ANOMALY_FEATURE_COLUMNS

    examples = pd.DataFrame([dict.fromkeys(ANOMALY_FEATURE_COLUMNS, 0)])
    assert np.isclose(
        verify_anomaly_model_round_trip(Detector(), Model(0.4), examples)["anomaly_score"].iloc[0],
        0.4,
    )
    with pytest.raises(RuntimeError, match="anomaly_score changed"):
        verify_anomaly_model_round_trip(Detector(), Model(0.5), examples)
