"""Only the assertion contract is exercised; no E2E pipeline or service is invoked."""

import pytest

from tests.infrastructure.evaluation._support import FOLLOW_UP_ACTIONS, STANDARD, action_check, recorder

TRIGGERS = [
    {"workflow": "EXPERT_REVIEW"},
    {"workflow": "INVESTIGATION"},
    {"anomalous": True},
    *[
        {"signals": [code]}
        for code in (
            "ANOMALOUS_PATTERN",
            "EXTREME_ANOMALY_PERCENTILE",
            "ADDITIONAL_QUOTE_DAMAGE",
            "UNMAPPED_QUOTE_ITEM",
            "RECENT_CLAIM_FREQUENCY",
        )
    ],
]


@pytest.mark.parametrize("trigger", TRIGGERS)
@pytest.mark.parametrize("action", [STANDARD, "NO_RECOMMENDATION"])
def test_observed_follow_up_cannot_be_ignored_even_at_normal_priority(trigger, action):
    check = action_check(recorder(action=action, **trigger))
    assert check.status == "FAIL"
    assert set(check.expected) == set(FOLLOW_UP_ACTIONS)


@pytest.mark.parametrize("action", FOLLOW_UP_ACTIONS)
@pytest.mark.parametrize("trigger", TRIGGERS)
def test_observed_follow_up_accepts_the_human_review_actions(trigger, action):
    assert action_check(recorder(action=action, **trigger)).status == "PASS"


@pytest.mark.parametrize("workflow", ["STANDARD", "FAST_TRACK"])
@pytest.mark.parametrize("action", [STANDARD, *FOLLOW_UP_ACTIONS])
def test_negative_real_models_do_not_inherit_a_deterministic_anomaly_label(workflow, action):
    assert action_check(recorder(action=action, workflow=workflow)).status == "PASS"


def test_real_fallback_is_explicit_and_does_not_automatically_allow_abstention():
    assert action_check(recorder(action="NO_RECOMMENDATION")).status == "FAIL"


def test_deterministic_actions_and_scenarios_without_a_real_policy_remain_strict():
    assert action_check(recorder(mode="deterministic")).status == "FAIL"
    value = recorder()
    value.scenario.real_agent_action_policy = None
    assert action_check(value).status == "FAIL"


@pytest.mark.parametrize("component", ["triage", "anomaly", "investigation"])
@pytest.mark.parametrize("status", ["SKIP", "FAIL", None])
def test_missing_model_inputs_cannot_establish_that_follow_up_is_unnecessary(component, status):
    value = recorder()
    if status is None:
        del value.observation.components[component]
    else:
        value.observation.components[component].status = status
    assert action_check(value).status == "SKIP"


@pytest.mark.parametrize("component", ["triage", "anomaly", "investigation"])
def test_missing_payload_is_not_a_negative_prediction(component):
    value = recorder()
    if component == "investigation":
        value.observation.investigation = None
    else:
        del value.observation.facts[component]
    assert action_check(value).status == "SKIP"


@pytest.mark.parametrize("flag", ["false", "true", 0, 1, None])
def test_anomaly_flags_must_be_observed_booleans(flag):
    assert action_check(recorder(anomalous=flag)).status == "SKIP"


def test_one_positive_observation_requires_follow_up_despite_another_missing_input():
    value = recorder(workflow="EXPERT_REVIEW")
    value.observation.components["anomaly"].status = "SKIP"
    assert action_check(value).status == "FAIL"


def test_failed_component_does_not_supply_a_stale_positive_prediction():
    value = recorder(workflow="EXPERT_REVIEW")
    value.observation.components["triage"].status = "FAIL"
    assert action_check(value).status == "SKIP"


@pytest.mark.parametrize("missing", ["history", "consistency", "features", "anomaly"])
def test_partial_investigation_does_not_supply_a_negative_conclusion(missing):
    value = recorder()
    value.observation.components["investigation"].metadata = {"partial": True, "unavailable_inputs": [missing]}
    assert action_check(value).status == "SKIP"
    value.observation.facts["triage"]["recommended_workflow"] = "EXPERT_REVIEW"
    assert action_check(value).status == "FAIL"


@pytest.mark.parametrize("vision_required, status", [(False, "PASS"), (True, "SKIP")])
def test_only_optional_vision_can_be_omitted_from_a_negative_conclusion(vision_required, status):
    value = recorder()
    value.scenario.vision_required = vision_required
    value.observation.components["investigation"].metadata = {"partial": True, "unavailable_inputs": ["vision"]}
    assert action_check(value).status == status


def test_partial_investigation_without_recorded_missing_inputs_remains_unavailable():
    value = recorder()
    value.observation.components["investigation"].metadata = {"partial": True}
    assert action_check(value).status == "SKIP"
