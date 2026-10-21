"""Injected evidence and unsafe structured decisions must fail closed."""

import pytest
from pipeline_support import failures_for, observation_for


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
