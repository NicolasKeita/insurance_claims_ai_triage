"""Recommendation."""

from typing import Literal

from pydantic import Field

from agent.config import CLAIMS_AGENT_PROMPT_VERSION, CLAIMS_AGENT_VERSION
from agent.models._base import RecommendedNextAction, StrictModel
from knowledge.models import KnowledgeCitation


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
