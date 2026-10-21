"""Evidence."""

from pydantic import Field

from agent.models._base import (
    AgentToolName,
    EvidenceSourceType,
    ObservationStatus,
    StrictModel,
)


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
