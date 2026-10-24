"""Image expectations can differ by backend without executing a scenario."""

import json

import pytest

from evaluation.assertions.facts import check_expected_facts
from evaluation.assertions.recording import AssertionRecorder
from evaluation.models import EvaluationScenario, PipelineObservation
from evaluation.scenarios import ROOT, load_scenarios


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


def non_detection_checks(*, mode="real", detections=(), images_present=True):
    selected = load_scenarios(scenario_ids=["visual_non_detection"])[0]
    observation = PipelineObservation(facts={
        "vision": {
            "aggregate_mapped_damage_types": [],
            "consistency": {
                "additional_visual_damage": [],
                "declared_damage_not_visually_confirmed": ["LEFT_HEADLIGHT"],
                "visually_supported_declared_damage": [],
                "visually_supported_quote_damage": [],
            },
            "images": [{"damage_assessment": {"detections": list(detections)}}] if images_present else [],
        },
        "investigation": {
            "policy_version": "investigation_policy_v3",
            "review_score": 99,
            "signals": [{"code": "ANOMALOUS_PATTERN"}],
        },
        "triage": {"recommended_workflow": "EXPERT_REVIEW"},
        "anomaly": {"is_anomalous": True},
        "history": {"previous_claim_count": 3},
    })
    recorder = AssertionRecorder(selected, observation, mode=mode)
    check_expected_facts(recorder)
    return selected, {item.name: item for item in recorder.checks}


def test_manifest_non_detection_requires_intact_image_and_unconfirmed_headlight():
    _, result = non_detection_checks()
    visual = {name: item for name, item in result.items() if name.startswith("fact:vision.")}
    assert "fact:vision.images" in visual
    assert "fact:vision.images.0.damage_assessment.detections" in visual
    assert "fact:vision.consistency.visually_supported_declared_damage" in visual
    assert "fact:vision.consistency.visually_supported_quote_damage" in visual
    assert all(item.status == "PASS" for item in visual.values())
    assert visual["fact:vision.consistency.declared_damage_not_visually_confirmed"].expected == ["LEFT_HEADLIGHT"]


def test_manifest_non_detection_rejects_unmapped_false_detection():
    _, result = non_detection_checks(detections=[{"damage_type": "SCRATCH"}])
    assert result["fact:vision.aggregate_mapped_damage_types"].status == "PASS"
    assert result["fact:vision.images.0.damage_assessment.detections"].status == "FAIL"


def test_manifest_non_detection_requires_an_image_even_when_no_parts_are_mapped():
    _, result = non_detection_checks(images_present=False)
    assert result["fact:vision.aggregate_mapped_damage_types"].status == "PASS"
    assert result["fact:vision.images"].status == "FAIL"
    assert result["fact:vision.images.0.damage_assessment.detections"].status == "FAIL"


def test_manifest_non_detection_preserves_original_deterministic_expectations():
    selected, result = non_detection_checks(mode="deterministic", images_present=False)
    legacy = json.loads((ROOT / "data/evaluation/e2e_v1/scenarios.json").read_text(encoding="utf-8"))
    previous = next(item for item in legacy["scenarios"] if item["scenario_id"] == selected.scenario_id)
    assert selected.expected_facts == previous["expected_facts"]
    assert selected.inputs == previous["inputs"]
    assert all(item.status == "PASS" for name, item in result.items() if name.startswith("fact:vision."))
    assert "fact:vision.images" not in result
    assert "fact:vision.images.0.damage_assessment.detections" not in result
    assert result["fact:investigation.review_score"].status == "FAIL"
    assert result["fact:investigation.signals"].status == "FAIL"


def test_manifest_non_detection_keeps_real_ml_and_history_findings_independent():
    _, result = non_detection_checks()
    assert result["fact:investigation.policy_version"].status == "PASS"
    assert "fact:investigation.review_score" not in result
    assert "fact:investigation.signals" not in result
    assert not any(name.startswith(("fact:triage.", "fact:anomaly.", "fact:history.")) for name in result)
