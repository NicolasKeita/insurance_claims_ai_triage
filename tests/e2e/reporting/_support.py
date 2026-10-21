"""Support."""

from evaluation.models import (
    EndToEndEvaluationReport,
    EvaluationAssertion,
    PipelineObservation,
    ScenarioEvaluationResult,
)


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
