"""Load the reviewable, versioned golden cases without running any services."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaluation.models import E2E_BENCHMARK_VERSION, EvaluationScenario

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS_PATH = ROOT / "data/evaluation/e2e_v1/scenarios.json"


class EvaluationBenchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_version: str
    scenarios: list[EvaluationScenario] = Field(min_length=1)

    @model_validator(mode="after")
    def supported_and_unique(self):
        if self.benchmark_version != E2E_BENCHMARK_VERSION:
            raise ValueError(
                f"Unsupported E2E benchmark version {self.benchmark_version!r}; expected {E2E_BENCHMARK_VERSION!r}"
            )
        ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("E2E scenario IDs must be unique")
        return self


def load_scenarios(
    path: str | Path | None = None,
    scenario_ids: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[EvaluationScenario]:
    """Validate the whole manifest, then select IDs in benchmark order.

    An unknown filter is an error: a typo must not look like a successful empty
    evaluation. Paths are resolved independently of the caller's directory.
    """
    benchmark_path = DEFAULT_SCENARIOS_PATH if path is None else Path(path)
    benchmark = EvaluationBenchmark.model_validate_json(benchmark_path.read_text(encoding="utf-8"))
    if scenario_ids is None:
        return benchmark.scenarios
    requested = set(scenario_ids)
    unknown = requested - {scenario.scenario_id for scenario in benchmark.scenarios}
    if unknown:
        raise ValueError(f"Unknown E2E scenario IDs: {', '.join(sorted(unknown))}")
    return [scenario for scenario in benchmark.scenarios if scenario.scenario_id in requested]
