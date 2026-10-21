"""Test assessments."""

import json
from uuid import uuid4

import pytest

from persistence.mappers import (
    anomaly_to_row,
    checked_triage,
    investigation_to_row,
    triage_to_row,
)
from tests.domain.persistence.mapping._support import (
    make_anomaly,
    make_investigation,
    make_triage,
)
from tests.domain.persistence.mapping._support import reference_claim as reference_claim


def test_assessment_mappers_keep_metadata_and_evidence():
    pk = uuid4()
    triage = triage_to_row(pk, make_triage())
    anomaly = anomaly_to_row(pk, make_anomaly())
    source = make_investigation()
    investigation = investigation_to_row(pk, source)
    assert triage.probabilities == make_triage().probabilities
    assert anomaly.unusual_features == make_anomaly().assessment.statistical_signals
    assert anomaly.reference_percentile == 0.995
    assert investigation.signals == source.model_dump(mode="json")["signals"]
    source.signals[0].evidence["unusual_features"].append("changed later")
    assert investigation.signals[0]["evidence"]["unusual_features"] == ["repair_amount: very_high"]
    json.dumps(investigation.signals, allow_nan=False)


@pytest.mark.parametrize(
    "probabilities",
    [
        {},
        {"STANDARD": 1.1, "FAST_TRACK": -0.1},
        {"STANDARD": 0.7, "FAST_TRACK": 0.1},
        {"OTHER": 1},
        {"STANDARD": float("nan")},
        {"STANDARD": 0.8, "FAST_TRACK": 0.2},
    ],
)
def test_incoherent_probabilities_rejected(probabilities):
    with pytest.raises(ValueError, match="Incoherent"):
        checked_triage(make_triage().model_copy(update={"probabilities": probabilities}))


def test_nonfinite_anomaly_rejected():
    result = make_anomaly()
    result.assessment.anomaly_score = float("nan")
    with pytest.raises(ValueError, match="finite"):
        anomaly_to_row(uuid4(), result)
