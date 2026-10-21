"""Test provenance."""

import pytest

from tests.e2e.deterministic._support import failures_for, golden_run, observation_for
from tests.e2e.deterministic._support import golden_run as golden_run


@pytest.mark.parametrize(
    "source, expected_failure",
    [
        ("documents", "provenance:documents"),
        ("anomaly", "provenance:anomaly_model"),
    ],
)
def test_missing_provenance_fails(golden_run, source, expected_failure):
    scenario, observation = observation_for(golden_run, "additional_quote_damage")
    if source == "documents":
        observation.provenance["documents"] = []
    else:
        observation.model_versions.pop("anomaly", None)
    assert expected_failure in failures_for(scenario, observation)


def test_missing_visual_model_provenance_fails(golden_run):
    scenario, observation = observation_for(golden_run, "visual_additional_damage")
    observation.vision["images"][0]["part_assessment"]["model_metadata"].pop("version")
    assert "provenance:vision" in failures_for(scenario, observation)
