"""Image expectations can differ by backend without executing a scenario."""

import pytest

from evaluation.assertions.facts import check_expected_facts
from evaluation.assertions.recording import AssertionRecorder
from evaluation.models import EvaluationScenario, PipelineObservation


def scenario():
    return EvaluationScenario(
        scenario_id="visual_controlled",
        description="Fixed geometry in the deterministic backend; annotated pixels in real inference.",
        claim_source="unused",
        expected_facts={
            "extraction.success": True,
            "vision.aggregate_mapped_damage_types": ["HOOD"],
            "vision.consistency.additional_visual_damage": ["HOOD"],
        },
        real_expected_facts={
            "vision.aggregate_mapped_damage_types": {"$set_equals": ["FRONT_BUMPER", "HOOD"]},
            "vision.consistency.additional_visual_damage": {"$set_equals": ["HOOD"]},
            "vision.consistency.visually_supported_declared_damage": {"$set_equals": ["FRONT_BUMPER"]},
        },
    )


def checks(mode, *, mapped, additional=("HOOD",), supported=()):
    observation = PipelineObservation(facts={
        "extraction": {"success": True},
        "vision": {
            "aggregate_mapped_damage_types": list(mapped),
            "consistency": {
                "additional_visual_damage": list(additional),
                "visually_supported_declared_damage": list(supported),
            },
        },
    })
    recorder = AssertionRecorder(scenario(), observation, mode=mode)
    check_expected_facts(recorder)
    return {item.name: item for item in recorder.checks}


def test_deterministic_facts_keep_the_original_geometry_contract():
    result = checks("deterministic", mapped=["HOOD"])
    assert len(result) == 3
    assert all(item.status == "PASS" for item in result.values())
    assert result["fact:vision.aggregate_mapped_damage_types"].expected == ["HOOD"]
    changed = checks("deterministic", mapped=["FRONT_BUMPER", "HOOD"])
    assert changed["fact:vision.aggregate_mapped_damage_types"].status == "FAIL"


def test_real_facts_require_both_annotated_parts_and_keep_common_facts():
    result = checks("real", mapped=["HOOD", "FRONT_BUMPER"], supported=["FRONT_BUMPER"])
    assert len(result) == 4
    assert all(item.status == "PASS" for item in result.values())
    assert result["fact:extraction.success"].expected is True
    assert result["fact:vision.aggregate_mapped_damage_types"].expected == {
        "$set_equals": ["FRONT_BUMPER", "HOOD"],
    }


@pytest.mark.parametrize("mapped,additional,supported", [
    (["HOOD"], ["HOOD"], []),
    (["FRONT_BUMPER"], [], ["FRONT_BUMPER"]),
    (["FRONT_BUMPER", "HOOD", "LEFT_HEADLIGHT"], ["HOOD", "LEFT_HEADLIGHT"], ["FRONT_BUMPER"]),
])
def test_real_facts_reject_missing_or_extra_damage(mapped, additional, supported):
    result = checks("real", mapped=mapped, additional=additional, supported=supported)
    assert result["fact:vision.aggregate_mapped_damage_types"].status == "FAIL"
    assert any(item.status == "FAIL" for item in result.values())


def test_real_overrides_do_not_mutate_the_deterministic_expectations():
    original = scenario()
    before = original.model_dump()
    for mode in ("real", "deterministic"):
        recorder = AssertionRecorder(original, PipelineObservation(), mode=mode)
        check_expected_facts(recorder)
    assert original.model_dump() == before


def test_scenarios_without_real_overrides_keep_their_existing_contract():
    original = EvaluationScenario(
        scenario_id="controlled",
        description="Existing scenario",
        claim_source="unused",
        expected_facts={"extraction.success": True},
    )
    assert original.real_expected_facts == {}
    observation = PipelineObservation(facts={"extraction": {"success": True}})
    for mode in ("real", "deterministic"):
        recorder = AssertionRecorder(original, observation, mode=mode)
        check_expected_facts(recorder)
        assert len(recorder.checks) == 1
        assert recorder.checks[0].status == "PASS"
