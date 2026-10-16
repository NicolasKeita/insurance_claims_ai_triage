"""Collect structured checks with a stable trace source per component."""

from dataclasses import dataclass
from typing import Any

from evaluation.models import EvaluationAssertion


@dataclass(frozen=True)
class Check:
    """One structured expectation and how it was actually observed."""

    name: str
    component: str
    expected: Any = None
    actual: Any = None
    passed: bool = False
    source: str | None = None
    blocked: bool = False


class AssertionRecorder:
    """Accumulate assertions for one scenario against one observation."""

    def __init__(self, scenario, observation, mode="deterministic"):
        self.scenario = scenario
        self.observation = observation
        self.mode = mode
        self.checks = []

    def source(self, component):
        return f"traces/{self.scenario.scenario_id}.json#{component}"

    def skipped(self, component):
        item = self.observation.components.get(component)
        return item is not None and item.status == "SKIP"

    def record(self, check: Check):
        status = "SKIP" if check.blocked else ("PASS" if check.passed else "FAIL")
        self.checks.append(
            EvaluationAssertion(
                name=check.name,
                component=check.component,
                status=status,
                expected=check.expected,
                actual=check.actual,
                source=check.source or self.source(check.component),
            )
        )