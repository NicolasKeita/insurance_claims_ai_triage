"""Helpers shared by the offline end-to-end contract tests."""

from evaluation.assertions import evaluate_assertions
from evaluation.models import (
    EndToEndEvaluationReport,
    EvaluationAssertion,
    PipelineObservation,
    ScenarioEvaluationResult,
)


def failures_for(scenario, observation):
    return {check.name for check in evaluate_assertions(scenario, observation) if check.status == "FAIL"}


def observation_for(golden_run, scenario_id):
    scenarios, report = golden_run
    observation = next(result.observation for result in report.results if result.scenario_id == scenario_id)
    return scenarios[scenario_id], observation.model_copy(deep=True)


def result(scenario_id, status, *, checks=None, components=None):
    return ScenarioEvaluationResult(
        scenario_id=scenario_id,
        status=status,
        assertions=checks or [],
        component_results=components or {},
        observation=PipelineObservation(
            model_versions={"anomaly": "model-v1"}, policy_versions={"investigation": "policy-v3"}
        ),
    )


def baseline_report():
    checks = [
        EvaluationAssertion(name=name, component=component, status="PASS", expected=True, actual=True)
        for name, component in [
            ("required_signal:ADDITIONAL_QUOTE_DAMAGE", "investigation"),
            ("forbidden_signal:VISUAL_ADDITIONAL_DAMAGE", "investigation"),
            ("rag:expected_sources", "retrieval"),
            ("rag:citation_validity", "retrieval"),
            ("agent:evidence_citations", "agent"),
        ]
    ]
    return EndToEndEvaluationReport(
        mode="real", full_stack_status="COMPLETE", results=[result("golden", "PASS", checks=checks)]
    )
