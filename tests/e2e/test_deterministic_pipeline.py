"""Fast offline system contracts using the same domain/application services."""

import pytest


@pytest.mark.parametrize(
    "scenario_id",
    [
        "simple_coherent_claim",
        "additional_quote_damage",
        "missing_documents",
        "recent_claim_history",
        "statistically_unusual_claim",
        "policy_answerable",
        "policy_insufficient_evidence",
        "agent_missing_tool_data",
        "visual_additional_damage",
        "visual_non_detection",
    ],
)
def test_golden_scenario_satisfies_all_mandatory_expectations(golden_run, scenario_id):
    _, report = golden_run
    result = next(result for result in report.results if result.scenario_id == scenario_id)
    failures = [
        check.model_dump(mode="json") for check in result.assertions if check.mandatory and check.status != "PASS"
    ]
    assert result.status == "PASS", {"scenario": scenario_id, "errors": result.errors, "assertions": failures}
    assert not failures


def test_visual_non_detection_adds_no_review_points_and_preserves_uncertainty(golden_run):
    _, report = golden_run
    result = next(result for result in report.results if result.scenario_id == "visual_non_detection")
    facts = result.observation.facts
    assert facts["vision"]["consistency"]["declared_damage_not_visually_confirmed"] == ["LEFT_HEADLIGHT"]
    assert facts["investigation"]["review_score"] == 0
    assert result.observation.signals == []


def test_pipeline_trace_records_component_outputs_and_sources(golden_run):
    _, report = golden_run
    result = next(result for result in report.results if result.scenario_id == "visual_additional_damage")
    trace = result.observation.trace
    completed = {step.component for step in trace}
    assert {"document", "consistency", "triage", "anomaly", "history", "investigation", "vision", "agent"} <= completed
    assert all(step.duration_seconds >= 0 for step in trace)
    assert result.observation.provenance["documents"]
    assert all("chain_of_thought" not in step.model_dump() for step in trace)
