"""Fast offline system contracts using the same domain/application services."""

import pytest

from evaluation.assertions import evaluate_assertions
from evaluation.pipeline import ClaimsEvaluationPipeline
from evaluation.runner import evaluate_scenarios
from evaluation.scenarios import load_scenarios


@pytest.fixture(scope="module")
def golden_run():
    scenarios = load_scenarios()
    report = evaluate_scenarios(scenarios, ClaimsEvaluationPipeline())
    return {scenario.scenario_id: scenario for scenario in scenarios}, report


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


def failures_for(scenario, observation):
    return {check.name for check in evaluate_assertions(scenario, observation) if check.status == "FAIL"}


def observation_for(golden_run, scenario_id):
    scenarios, report = golden_run
    observation = next(result.observation for result in report.results if result.scenario_id == scenario_id)
    return scenarios[scenario_id], observation.model_copy(deep=True)


def test_unknown_agent_evidence_id_fails_the_scenario(golden_run):
    scenario, observation = observation_for(golden_run, "simple_coherent_claim")
    recommendation = observation.agent["recommendation"]
    recommendation["rationale"][0]["evidence_ids"].append("invented:evidence")
    recommendation["evidence_ids"].append("invented:evidence")
    assert "agent:evidence_citations" in failures_for(scenario, observation)


def test_unknown_rag_citation_fails_even_with_a_well_formed_answer(golden_run):
    scenario, observation = observation_for(golden_run, "policy_answerable")
    observation.rag["answer"]["citations"][0]["chunk_id"] = "00000000-0000-0000-0000-000000000000"
    assert "rag:citation_validity" in failures_for(scenario, observation)


def test_known_chunk_id_cannot_hide_fabricated_policy_metadata(golden_run):
    scenario, observation = observation_for(golden_run, "policy_answerable")
    observation.rag["answer"]["citations"][0]["section"] = "Invented Exclusion"
    assert "rag:citation_validity" in failures_for(scenario, observation)


def test_a_missing_required_rag_answer_does_not_pass_as_empty_context(golden_run):
    scenario, observation = observation_for(golden_run, "policy_answerable")
    observation.rag = None
    observation.facts.pop("rag")
    assert "fact:rag.answer.insufficient_evidence" in failures_for(scenario, observation)


@pytest.mark.parametrize("decision", ["FRAUD_CONFIRMED", "APPROVE_CLAIM", "DENY_CLAIM", "PAYMENT_AUTHORIZED"])
def test_forbidden_structured_agent_decisions_fail_schema_and_safety(golden_run, decision):
    scenario, observation = observation_for(golden_run, "simple_coherent_claim")
    observation.agent["recommendation"]["recommended_next_action"] = decision
    failures = failures_for(scenario, observation)
    assert {"schema:agent", "safety:forbidden_actions"} <= failures


def test_forbidden_investigation_decision_is_rejected(golden_run):
    scenario, observation = observation_for(golden_run, "simple_coherent_claim")
    observation.investigation["decision"] = "DENY_CLAIM"
    assert {"schema:investigation", "safety:forbidden_actions"} <= failures_for(scenario, observation)


def test_a_malformed_final_schema_fails_without_a_forbidden_action(golden_run):
    scenario, observation = observation_for(golden_run, "simple_coherent_claim")
    del observation.agent["recommendation"]["rationale"]
    assert "schema:agent" in failures_for(scenario, observation)


@pytest.mark.parametrize(
    "field, wrong_taxonomy",
    [
        ("visual_damage_type", "FRONT_BUMPER"),
        ("vehicle_part", "SCRATCH"),
        ("mapped_damage_type", "SCRATCH"),
    ],
)
def test_visual_morphology_vehicle_parts_and_business_damage_remain_separate(golden_run, field, wrong_taxonomy):
    scenario, observation = observation_for(golden_run, "visual_additional_damage")
    observation.vision["images"][0]["evidence"][0][field] = wrong_taxonomy
    assert "schema:vision" in failures_for(scenario, observation)


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
