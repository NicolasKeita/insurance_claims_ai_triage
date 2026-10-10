"""Closed application contracts: the LLM cannot add capabilities or change scope."""

from datetime import datetime
from enum import StrEnum
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.config import CLAIMS_AGENT_PROMPT_VERSION, CLAIMS_AGENT_VERSION
from knowledge.models import KnowledgeCitation


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AgentToolName(StrEnum):
    GET_CLAIM = "GET_CLAIM"
    GET_HISTORY = "GET_HISTORY"
    GET_TRIAGE = "GET_TRIAGE"
    GET_ANOMALY = "GET_ANOMALY"
    GET_INVESTIGATION = "GET_INVESTIGATION"
    FIND_SIMILAR_CLAIMS = "FIND_SIMILAR_CLAIMS"
    SEARCH_POLICY = "SEARCH_POLICY"
    SEARCH_PROCEDURE = "SEARCH_PROCEDURE"


class AgentAction(StrEnum):
    GET_CLAIM = "GET_CLAIM"
    GET_HISTORY = "GET_HISTORY"
    GET_TRIAGE = "GET_TRIAGE"
    GET_ANOMALY = "GET_ANOMALY"
    GET_INVESTIGATION = "GET_INVESTIGATION"
    FIND_SIMILAR_CLAIMS = "FIND_SIMILAR_CLAIMS"
    SEARCH_POLICY = "SEARCH_POLICY"
    SEARCH_PROCEDURE = "SEARCH_PROCEDURE"
    FINISH = "FINISH"


class EmptyToolInput(StrictModel):
    pass


class SimilarClaimsInput(StrictModel):
    limit: int = Field(default=3, ge=1, le=5, strict=True)


class KnowledgeSearchInput(StrictModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=3, ge=1, le=5, strict=True)


class AgentPlanStep(StrictModel):
    action: AgentAction
    reason: str = Field(min_length=1, max_length=300,
                        description="Short justification for the selected action, not private reasoning.")
    arguments: EmptyToolInput | SimilarClaimsInput | KnowledgeSearchInput = Field(
        description="Required for every action: {} for GET/FINISH, limit for similar, query and limit for search.")

    def validated_arguments(self):
        schema = EmptyToolInput
        if self.action == AgentAction.FIND_SIMILAR_CLAIMS:
            schema = SimilarClaimsInput
        elif self.action in (AgentAction.SEARCH_POLICY, AgentAction.SEARCH_PROCEDURE):
            schema = KnowledgeSearchInput
        return schema.model_validate(self.arguments.model_dump())

    @model_validator(mode="after")
    def action_arguments(self):
        self.arguments = self.validated_arguments()
        return self

    @property
    def tool_name(self) -> AgentToolName | None:
        return None if self.action == AgentAction.FINISH else AgentToolName(self.action.value)


class ObservationStatus(StrEnum):
    OK = "OK"
    NOT_FOUND = "NOT_FOUND"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    DUPLICATE_BLOCKED = "DUPLICATE_BLOCKED"


class EvidenceSourceType(StrEnum):
    CLAIM_DATA = "CLAIM_DATA"
    HISTORY = "HISTORY"
    TRIAGE_ASSESSMENT = "TRIAGE_ASSESSMENT"
    ANOMALY_ASSESSMENT = "ANOMALY_ASSESSMENT"
    INVESTIGATION_ASSESSMENT = "INVESTIGATION_ASSESSMENT"
    SIMILAR_CLAIM = "SIMILAR_CLAIM"
    POLICY_CHUNK = "POLICY_CHUNK"
    PROCEDURE_CHUNK = "PROCEDURE_CHUNK"


class EvidenceItem(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=300)
    source_type: EvidenceSourceType
    source_reference: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=4000)
    structured_data: dict | None = None


class ToolObservation(StrictModel):
    tool_name: AgentToolName
    status: ObservationStatus
    data: dict | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list, max_length=5)
    message: str | None = Field(default=None, max_length=500)


class RecommendedNextAction(StrEnum):
    CONTINUE_STANDARD_REVIEW = "CONTINUE_STANDARD_REVIEW"
    REQUEST_ADDITIONAL_INFORMATION = "REQUEST_ADDITIONAL_INFORMATION"
    REFER_TO_EXPERT = "REFER_TO_EXPERT"
    REFER_FOR_INVESTIGATION_REVIEW = "REFER_FOR_INVESTIGATION_REVIEW"
    NO_RECOMMENDATION = "NO_RECOMMENDATION"


class RecommendationRationale(StrictModel):
    statement: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=40)


class AgentRecommendation(StrictModel):
    case_summary: str = Field(min_length=1, max_length=3000)
    recommended_next_action: RecommendedNextAction
    rationale: list[RecommendationRationale] = Field(max_length=12,
        description="Required structured rationale; every factual statement cites collected evidence.")
    uncertainties: list[str] = Field(max_length=20, description="Required list; empty only if no uncertainties.")
    evidence_ids: list[str] = Field(max_length=40,
        description="Required union of exact collected IDs supporting rationale.")
    human_review_required: Literal[True] = True
    # Historical audit reads must remain valid after a future version bump.
    # The graph enforces the current versions on every newly generated answer.
    agent_version: str = Field(default=CLAIMS_AGENT_VERSION, min_length=1, max_length=100)
    prompt_version: str = Field(default=CLAIMS_AGENT_PROMPT_VERSION, min_length=1, max_length=100)
    # Resolved by application from collected evidence, never accepted from the LLM.
    citations: list[KnowledgeCitation] = Field(default_factory=list, max_length=40)


class RunStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


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
