"""Shared golden run for the offline end-to-end contracts."""

import pytest

from evaluation.pipeline import ClaimsEvaluationPipeline
from evaluation.runner import evaluate_scenarios
from evaluation.scenarios import load_scenarios


@pytest.fixture(scope="session")
def golden_run():
    scenarios = load_scenarios()
    report = evaluate_scenarios(scenarios, ClaimsEvaluationPipeline())
    return {scenario.scenario_id: scenario for scenario in scenarios}, report
