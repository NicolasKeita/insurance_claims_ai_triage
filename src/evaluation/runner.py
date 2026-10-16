"""Execute every compatible case and retain failures with useful diagnostics."""

from time import perf_counter

from evaluation.assertions import evaluate_assertions
from evaluation.metrics import aggregate_metrics
from evaluation.models import (ComponentResult, EndToEndEvaluationReport, EvaluationAssertion,
                               PipelineObservation, ScenarioEvaluationResult)


def evaluate_scenarios(scenarios, pipeline, *, mode="deterministic", preflight=None, fail_fast=False):
    results = []
    for scenario in scenarios:
        start = perf_counter()
        observation = None
        try:
            observation = PipelineObservation.model_validate(pipeline.run(scenario))
            assertions = evaluate_assertions(scenario, observation, mode=mode)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            if observation is None:
                observation = PipelineObservation()
            observation.errors.append(message)
            observation.components["pipeline"] = ComponentResult(status="FAIL", reason=message)
            assertions = [EvaluationAssertion(name="pipeline:execution", component="pipeline", status="FAIL",
                          expected="successful structured observation", actual=message, source=f"traces/{scenario.scenario_id}.json")]
        failed = any(a.mandatory and a.status == "FAIL" for a in assertions)
        failed |= any(c.status == "FAIL" for c in observation.components.values()) or bool(observation.errors)
        blocked = any(a.mandatory and a.status == "SKIP" for a in assertions)
        status = "FAIL" if failed else "SKIP" if blocked else "PASS"
        timings = {name: value.duration_seconds for name, value in observation.components.items()}
        timings["total"] = perf_counter() - start
        ml = [timings[k] for k in ("triage", "anomaly") if k in timings and observation.components[k].status != "SKIP"]
        if ml:
            timings["ml"] = sum(ml)
        results.append(ScenarioEvaluationResult(scenario_id=scenario.scenario_id, status=status,
            assertions=assertions, component_results=observation.components, timings=timings,
            errors=observation.errors, trace_reference=f"traces/{scenario.scenario_id}.json", observation=observation))
        if failed and fail_fast:
            break
    complete = (bool(results) and len(results) == len(scenarios) and all(r.status == "PASS" for r in results)
                and all(c.status == "PASS" or c.metadata.get("applicable") is False
                        for r in results for c in r.component_results.values()))
    return EndToEndEvaluationReport(mode=mode, results=results, preflight=preflight or [],
        full_stack_status="COMPLETE" if complete else "PARTIAL", metrics=aggregate_metrics(results))
