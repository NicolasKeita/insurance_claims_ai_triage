"""Closed base contracts and bounded tool inputs."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AgentToolName(StrEnum):
    GET_CLAIM = "GET_CLAIM"
    GET_HISTORY = "GET_HISTORY"
    GET_TRIAGE = "GET_TRIAGE"
    GET_ANOMALY = "GET_ANOMALY"
    GET_INVESTIGATION = "GET_INVESTIGATION"
    GET_VISUAL_ASSESSMENT = "GET_VISUAL_ASSESSMENT"
    FIND_SIMILAR_CLAIMS = "FIND_SIMILAR_CLAIMS"
    SEARCH_POLICY = "SEARCH_POLICY"
    SEARCH_PROCEDURE = "SEARCH_PROCEDURE"


class AgentAction(StrEnum):
    GET_CLAIM = "GET_CLAIM"
    GET_HISTORY = "GET_HISTORY"
    GET_TRIAGE = "GET_TRIAGE"
    GET_ANOMALY = "GET_ANOMALY"
    GET_INVESTIGATION = "GET_INVESTIGATION"
    GET_VISUAL_ASSESSMENT = "GET_VISUAL_ASSESSMENT"
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
