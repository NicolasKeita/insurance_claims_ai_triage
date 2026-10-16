"""Real-run missingness is tested offline without executing a real stack."""

import pytest

from evaluation import fakes
from evaluation.preflight import PreflightReport, RealEvaluationConfig
from evaluation.real import RealClaimsEvaluationPipeline
from evaluation.runner import evaluate_scenarios
from evaluation.scenarios import load_scenarios


@pytest.mark.parametrize("unavailable_policy, expected_status", [("SKIP", "SKIP"), ("FAIL", "FAIL")])
def test_unavailable_real_dependencies_never_become_fake_predictions(
    tmp_path,
    monkeypatch,
    unavailable_policy,
    expected_status,
):
    def forbid_fake(*args, **kwargs):
        raise AssertionError("A real evaluation called a deterministic predictor")

    monkeypatch.setattr(fakes.FakeTriagePredictor, "predict", forbid_fake)
    monkeypatch.setattr(fakes.FakeAnomalyPredictor, "assess", forbid_fake)
    preflight = PreflightReport(
        rows=[
            {
                "component": component,
                "status": unavailable_policy,
                "reason": "Controlled unavailable dependency",
                "version": None,
            }
            for component in [
                "postgresql",
                "qdrant",
                "knowledge_collections",
                "similar_claims_collection",
                "embeddings",
                "ollama",
                "triage",
                "anomaly",
                "damage_vision",
                "part_vision",
            ]
        ]
    )
    config = RealEvaluationConfig(fixture_dir=tmp_path / "fixtures", dependency_unavailable=unavailable_policy)
    pipeline = RealClaimsEvaluationPipeline(config, preflight)
    report = evaluate_scenarios(
        load_scenarios(scenario_ids=["policy_answerable"]), pipeline, mode="real", preflight=preflight.rows
    )
    assert report.full_stack_status == "PARTIAL"
    assert report.results[0].status == expected_status
    observed = report.results[0].observation
    assert observed.components["document"].status == "PASS"
    assert observed.facts["extraction"]["success"] is True
    assert observed.components["triage"].status == unavailable_policy
    assert observed.components["anomaly"].status == unavailable_policy
    assert observed.components["retrieval"].status == unavailable_policy
    assert "triage" not in observed.facts and "anomaly" not in observed.facts
    assert observed.model_versions == {}
    assert observed.agent is None and observed.rag is None
    if unavailable_policy == "SKIP":
        assert not [check for check in report.results[0].assertions if check.status == "FAIL"]
        assert report.metrics["scenario_pass_rate"] is None


def test_fake_visual_golden_labels_are_not_reused_as_real_image_ground_truth(tmp_path):
    config = RealEvaluationConfig(fixture_dir=tmp_path / "fixtures")
    pipeline = RealClaimsEvaluationPipeline(config, PreflightReport())
    report = evaluate_scenarios(load_scenarios(scenario_ids=["visual_additional_damage"]), pipeline, mode="real")
    observed = report.results[0].observation
    assert report.results[0].status == "SKIP"
    assert observed.components["vision"].status == "SKIP"
    assert "no matching real image ground truth" in observed.components["vision"].reason
    assert observed.vision is None
    assert "VISUAL_ADDITIONAL_DAMAGE" not in observed.signals
