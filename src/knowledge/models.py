from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class KnowledgeSourceType(StrEnum):
    POLICY = "POLICY"
    PROCEDURE = "PROCEDURE"


class KnowledgeSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    source_id: str = Field(min_length=1)
    source_type: KnowledgeSourceType
    title: str = Field(min_length=1)
    version: str = Field(min_length=1)
    product: str | None = None
    language: str = Field(min_length=2)
    effective_from: date | None = None
    effective_to: date | None = None
    file_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_interval(self):
        if self.effective_from and self.effective_to and self.effective_from >= self.effective_to:
            raise ValueError("effective_to must be after effective_from (exclusive end)")
        return self


class KnowledgeChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    chunk_id: str
    source_id: str
    source_type: KnowledgeSourceType
    source_version: str
    title: str
    section: str
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    product: str | None
    language: str
    effective_from: date | None
    effective_to: date | None
    chunking_version: str


class RetrievedKnowledgeChunk(KnowledgeChunk):
    similarity_score: float = Field(allow_inf_nan=False)


class KnowledgeSearchResponse(BaseModel):
    results: list[RetrievedKnowledgeChunk]


class GroundedLlmOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    answer: str = Field(min_length=1, max_length=12000)
    cited_chunk_ids: list[str] = Field(max_length=20)
    insufficient_evidence: StrictBool = Field(
        description="True only when the retrieved sources lack information needed to answer the question; false when they directly answer it within the demo.")


class KnowledgeCitation(BaseModel):
    chunk_id: str
    source_id: str
    source_version: str
    title: str
    section: str


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[KnowledgeCitation]
    insufficient_evidence: bool


class GroundingError(ValueError):
    """A structured answer violates the retrieved citation boundary."""


def applicable_on(chunk: KnowledgeChunk | KnowledgeSource, as_of_date: date | None) -> bool:
    """Open null bounds; inclusive start, exclusive end: [from, to)."""
    return as_of_date is None or (
        (chunk.effective_from is None or chunk.effective_from <= as_of_date)
        and (chunk.effective_to is None or as_of_date < chunk.effective_to)
    )
