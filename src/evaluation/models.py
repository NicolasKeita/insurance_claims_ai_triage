"""Serializable benchmark contracts; traces contain observations, never reasoning."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from evaluation import E2E_BENCHMARK_VERSION


class EvaluationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationScenario(EvaluationModel):
    scenario_id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    description: str
    claim_source: str
    expected_facts: dict[str, Any] = Field(default_factory=dict)
    expected_signals: list[str] = Field(default_factory=list)
    forbidden_signals: list[str] = Field(default_factory=list)
    acceptable_agent_actions: list[str] = Field(default_factory=list)
    expected_knowledge_sources: list[str] = Field(default_factory=list)
    expected_uncertainties: list[str] = Field(default_factory=list)
    vision_required: bool = False
    infrastructure_requirements: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    files: list[str] = Field(default_factory=list)
    external_assets: list[dict[str, Any]] = Field(default_factory=list)


class ComponentResult(EvaluationModel):
    status: EvaluationStatus
    duration_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(EvaluationModel):
    component: str
    status: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    source_references: list[str] = Field(default_factory=list)


class PipelineObservation(EvaluationModel):
    facts: dict[str, Any] = Field(default_factory=dict)
    signals: list[str] = Field(default_factory=list)
    agent: dict | None = None
    investigation: dict | None = None
    rag: dict | None = None
    vision: dict | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    components: dict[str, ComponentResult] = Field(default_factory=dict)
    trace: list[TraceEvent] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)
    policy_versions: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class EvaluationAssertion(EvaluationModel):
    name: str
    component: str
    status: EvaluationStatus
    expected: Any = None
    actual: Any = None
    source: str | None = None
    message: str | None = None
    mandatory: bool = True


class ScenarioEvaluationResult(EvaluationModel):
    scenario_id: str
    status: EvaluationStatus
    assertions: list[EvaluationAssertion] = Field(default_factory=list)
    component_results: dict[str, ComponentResult] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    trace_reference: str | None = None
    observation: PipelineObservation = Field(default_factory=PipelineObservation)


class EndToEndEvaluationReport(EvaluationModel):
    benchmark_version: str = E2E_BENCHMARK_VERSION
    mode: Literal["deterministic", "real"]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    full_stack_status: Literal["COMPLETE", "PARTIAL"] = "PARTIAL"
    results: list[ScenarioEvaluationResult] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    preflight: list[dict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: [
        "The benchmark validates expected behavior on controlled demo scenarios. It is not a measurement of real-world insurance decision accuracy.",
        "Synthetic policies/procedures and synthetic training data for some ML components.",
        "CarDD HF mirror custom test split; possible CV domain shift; small local LLM limitations.",
    ])
