"""Execute every compatible case and retain failures with useful diagnostics."""

from time import perf_counter

from evaluation.assertions import evaluate_assertions
from evaluation.metrics import aggregate_metrics
from evaluation.models import (
    ComponentResult,
    EndToEndEvaluationReport,
    EvaluationAssertion,
    PipelineObservation,
    ScenarioEvaluationResult,
)

ML_COMPONENTS = ("triage", "anomaly")


def _pipeline_failure(scenario, observation, error):
    message = f"{type(error).__name__}: {error}"
    observation.errors.append(message)
    observation.components["pipeline"] = ComponentResult(status="FAIL", reason=message)
    return [
        EvaluationAssertion(
            name="pipeline:execution",
            component="pipeline",
            status="FAIL",
            expected="successful structured observation",
            actual=message,
            source=f"traces/{scenario.scenario_id}.json",
        )
    ]


def _observe(pipeline, scenario, mode):
    observation = None
    try:
        observation = PipelineObservation.model_validate(pipeline.run(scenario))
        assertions = evaluate_assertions(scenario, observation, mode=mode)
    except Exception as error:
        if observation is None:
            observation = PipelineObservation()
        assertions = _pipeline_failure(scenario, observation, error)
    return observation, assertions


def _timings(observation, start):
    timings = {name: value.duration_seconds for name, value in observation.components.items()}
    timings["total"] = perf_counter() - start
    measured = [
        timings[name] for name in ML_COMPONENTS if name in timings and observation.components[name].status != "SKIP"
    ]
    if measured:
        timings["ml"] = sum(measured)
    return timings


def _scenario_status(observation, assertions):
    failed = any(check.mandatory and check.status == "FAIL" for check in assertions)
    failed |= any(component.status == "FAIL" for component in observation.components.values())
    failed |= bool(observation.errors)
    blocked = any(check.mandatory and check.status == "SKIP" for check in assertions)
    return "FAIL" if failed else "SKIP" if blocked else "PASS"


def _evaluate_scenario(scenario, pipeline, mode):
    start = perf_counter()
    observation, assertions = _observe(pipeline, scenario, mode)
    return ScenarioEvaluationResult(
        scenario_id=scenario.scenario_id,
        status=_scenario_status(observation, assertions),
        assertions=assertions,
        component_results=observation.components,
        timings=_timings(observation, start),
        errors=observation.errors,
        trace_reference=f"traces/{scenario.scenario_id}.json",
        observation=observation,
    )


def _stack_complete(results, scenarios):
    return (
        bool(results)
        and len(results) == len(scenarios)
        and all(result.status == "PASS" for result in results)
        and all(
            component.status == "PASS" or component.metadata.get("applicable") is False
            for result in results
            for component in result.component_results.values()
        )
    )


def evaluate_scenarios(scenarios, pipeline, *, mode="deterministic", preflight=None, fail_fast=False):
    """Run every compatible scenario and retain failures with diagnostics."""
    results = []
    for scenario in scenarios:
        result = _evaluate_scenario(scenario, pipeline, mode)
        results.append(result)
        if result.status == "FAIL" and fail_fast:
            break
    return EndToEndEvaluationReport(
        mode=mode,
        results=results,
        preflight=preflight or [],
        full_stack_status="COMPLETE" if _stack_complete(results, scenarios) else "PARTIAL",
        metrics=aggregate_metrics(results),
    )