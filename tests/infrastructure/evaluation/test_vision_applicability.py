"""Check disabled vision applicability without executing a pipeline or models."""

from types import SimpleNamespace

import pytest

from evaluation.models import ScenarioEvaluationResult
from evaluation.real.context import ComponentRecorder, RealRun
from evaluation.real.stages.vision import VisionSteps
from evaluation.runner import _stack_complete
from evaluation.scenarios import load_scenarios


class DisabledVision(VisionSteps, ComponentRecorder):
    config = SimpleNamespace(include_vision=False, dependency_unavailable="SKIP")


@pytest.mark.parametrize("scenario", load_scenarios(), ids=lambda item: item.scenario_id)
def test_disabled_vision_only_becomes_optional_for_non_visual_scenarios(scenario):
    run = RealRun(scenario=scenario)
    DisabledVision().assess_vision(run)
    component = run.observed.components["vision"]
    assert component.status == "SKIP"
    result = ScenarioEvaluationResult(
        scenario_id=scenario.scenario_id,
        status="PASS",
        component_results=run.observed.components,
    )
    if scenario.vision_required:
        assert component.metadata.get("applicable") is not False
        assert not _stack_complete([result], [scenario])
    else:
        assert component.metadata["applicable"] is False
        assert _stack_complete([result], [scenario])
