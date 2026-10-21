"""Support."""

import pytest

from evaluation.assertions import evaluate_assertions
from evaluation.pipeline import ClaimsEvaluationPipeline
from evaluation.runner import evaluate_scenarios
from evaluation.scenarios import load_scenarios


@pytest.fixture(scope="module")
def golden_run():
    scenarios = load_scenarios()
    report = evaluate_scenarios(scenarios, ClaimsEvaluationPipeline())
    return {scenario.scenario_id: scenario for scenario in scenarios}, report


def failures_for(scenario, observation):
    return {check.name for check in evaluate_assertions(scenario, observation) if check.status == "FAIL"}


def observation_for(golden_run, scenario_id):
    scenarios, report = golden_run
    observation = next(result.observation for result in report.results if result.scenario_id == scenario_id)
    return scenarios[scenario_id], observation.model_copy(deep=True)
