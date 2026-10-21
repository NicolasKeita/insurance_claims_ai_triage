"""Test contracts."""

import pytest

from tests.e2e.deterministic._support import failures_for, golden_run, observation_for
from tests.e2e.deterministic._support import golden_run as golden_run


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
