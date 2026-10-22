"""Attach per-operation monitoring to real scenarios without changing execution order."""

from evaluation.real.progress.monitor import ScenarioProgress
from evaluation.real.progress.reference import previous_timings


def run_with_progress(pipeline, scenario, output_dir):
    if output_dir is None or not hasattr(pipeline, "progress"):
        return pipeline.run(scenario)
    model = pipeline.resources.get("llm_model")
    reference = previous_timings(output_dir, scenario.scenario_id, model)
    previous = pipeline.progress
    with ScenarioProgress(output_dir, scenario.scenario_id, reference) as monitor:
        pipeline.progress = monitor
        try:
            result = pipeline.run(scenario)
            monitor.complete(result)
            return result
        finally:
            pipeline.progress = previous
