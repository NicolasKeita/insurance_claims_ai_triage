"""Tool observations, evidence items and the recommendation contract."""

from enum import StrEnum
from typing import Literal

from pydantic import Field

from agent.config import CLAIMS_AGENT_PROMPT_VERSION, CLAIMS_AGENT_VERSION
from agent.models.base import AgentToolName, StrictModel
from knowledge.models import KnowledgeCitation


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
    VISUAL_ASSESSMENT = "VISUAL_ASSESSMENT"
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
    rationale: list[RecommendationRationale] = Field(
        max_length=12, description="Required structured rationale; every factual statement cites collected evidence."
    )
    uncertainties: list[str] = Field(max_length=20, description="Required list; empty only if no uncertainties.")
    evidence_ids: list[str] = Field(
        max_length=40, description="Required union of exact collected IDs supporting rationale."
    )
    human_review_required: Literal[True] = True
    # Historical audit reads must remain valid after a future version bump.
    # The graph enforces the current versions on every newly generated answer.
    agent_version: str = Field(default=CLAIMS_AGENT_VERSION, min_length=1, max_length=100)
    prompt_version: str = Field(default=CLAIMS_AGENT_PROMPT_VERSION, min_length=1, max_length=100)
    # Resolved by application from collected evidence, never accepted from the LLM.
    citations: list[KnowledgeCitation] = Field(default_factory=list, max_length=40)
