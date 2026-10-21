"""Closed interface shared by graph adapters and test doubles."""

from typing import Protocol

from agent.models import AgentPlanStep, ToolObservation


class AgentTools(Protocol):
    def require_claim(self, claim_id: str) -> None: ...
    def execute(self, claim_id: str, step: AgentPlanStep) -> ToolObservation: ...
