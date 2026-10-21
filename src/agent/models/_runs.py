"""Runs."""

from datetime import datetime
from typing import TypedDict

from pydantic import Field, model_validator

from agent.config import CLAIMS_AGENT_PROMPT_VERSION, CLAIMS_AGENT_VERSION
from agent.models._base import (
    AgentAction,
    AgentToolName,
    ObservationStatus,
    RunStatus,
    StrictModel,
)
from agent.models._evidence import EvidenceItem
from agent.models._recommendation import AgentRecommendation


class AgentTraceStep(StrictModel):
    iteration: int = Field(ge=1, le=9)
    planner_action: AgentAction
    planner_reason: str = Field(min_length=1, max_length=300)
    tool_invoked: AgentToolName | None = None
    tool_args: dict = Field(default_factory=dict)
    observation_status: ObservationStatus | None = None
    evidence_ids_added: list[str] = Field(default_factory=list, max_length=5)


class AgentRunResult(StrictModel):
    run_id: str
    claim_id: str
    objective: str = Field(min_length=1, max_length=2000)
    status: RunStatus
    agent_version: str = CLAIMS_AGENT_VERSION
    prompt_version: str = CLAIMS_AGENT_PROMPT_VERSION
    llm_model: str | None = None
    recommendation: AgentRecommendation | None = None
    tools_used: list[AgentToolName] = Field(default_factory=list, max_length=8)
    iteration_count: int = Field(ge=0, le=8)
    tool_call_count: int = Field(ge=0, le=8)
    evidence: list[EvidenceItem] = Field(default_factory=list, max_length=40)
    trace: list[AgentTraceStep] = Field(default_factory=list, max_length=9)
    created_at: datetime
    completed_at: datetime
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)
    error: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def terminal_result(self):
        if self.status == RunStatus.RUNNING:
            raise ValueError("Only terminal runs may be returned or persisted")
        if self.status == RunStatus.COMPLETED and (self.recommendation is None or self.error is not None):
            raise ValueError("A completed run requires a recommendation and no error")
        if self.status == RunStatus.FAILED and (self.recommendation is not None or not self.error):
            raise ValueError("A failed run requires a sanitized error and no recommendation")
        return self


class AgentState(TypedDict):
    run_id: str
    claim_id: str
    objective: str
    iteration: int
    max_iterations: int
    plan_history: list[dict]
    observations: list[dict]
    evidence_registry: dict[str, dict]
    current_plan: dict | None
    invocation_counts: dict[str, int]
    trace: list[dict]
    recommendation: dict | None
    status: str
    error: str | None
    failure_kind: str | None
    stop_reason: str | None
    tool_call_count: int
