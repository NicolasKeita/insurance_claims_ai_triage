"""Test grounding."""

from tests.e2e.deterministic._support import failures_for, golden_run, observation_for
from tests.e2e.deterministic._support import golden_run as golden_run


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
