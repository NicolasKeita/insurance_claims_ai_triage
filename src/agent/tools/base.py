"""Bounded, claim-scoped read adapters around the existing repositories/services.

These adapters never compute or persist a business assessment. New SQL sessions
close before embedding/retrieval calls; unavailable assessments stay unavailable.
"""

from collections.abc import Callable
from typing import Protocol

from agent.models import (
    AgentPlanStep,
    AgentToolName,
    EvidenceItem,
    KnowledgeSearchInput,
    ObservationStatus,
    SimilarClaimsInput,
    ToolObservation,
)
from agent.tools.assessments import ASSESSMENT_TOOLS, AssessmentObservations
from agent.tools.knowledge import KnowledgeObservations
from agent.tools.observations import ClaimObservations
from agent.tools.similar import SimilarObservations
from claims.models import Claim
from persistence.repositories import ClaimNotFoundError, ClaimRepository


class AgentTools(Protocol):
    def require_claim(self, claim_id: str) -> None: ...
    def execute(self, claim_id: str, step: AgentPlanStep) -> ToolObservation: ...


class ReadOnlyAgentTools(ClaimObservations, AssessmentObservations, SimilarObservations, KnowledgeObservations):
    def __init__(
        self,
        factory,
        similar_service=None,
        knowledge_retriever=None,
        *,
        similar_service_loader: Callable | None = None,
        knowledge_retriever_loader: Callable | None = None,
    ):
        self.factory = factory
        self.similar_service = similar_service
        self.knowledge_retriever = knowledge_retriever
        self.similar_service_loader = similar_service_loader
        self.knowledge_retriever_loader = knowledge_retriever_loader

    def close(self) -> None:
        """Release standalone retrieval resources; API lifespan owns its caches."""
        seen = set()
        for service in (self.similar_service, self.knowledge_retriever):
            if service is not None and id(service) not in seen:
                seen.add(id(service))
                if hasattr(service, "close"):
                    service.close()

    def require_claim(self, claim_id: str) -> None:
        with self.factory() as session:
            ClaimRepository(session).require_pk(claim_id)

    def _claim(self, claim_id: str) -> Claim:
        with self.factory() as session:
            claim = ClaimRepository(session).get_by_claim_id(claim_id)
            if claim is None:
                raise ClaimNotFoundError(f"Claim {claim_id} does not exist")
            return claim

    @staticmethod
    def _missing(tool: AgentToolName, message: str, *, unavailable=False):
        return ToolObservation(
            tool_name=tool,
            status=ObservationStatus.NOT_AVAILABLE if unavailable else ObservationStatus.NOT_FOUND,
            data=None,
            evidence=[],
            message=message,
        )

    @staticmethod
    def _observation(tool: AgentToolName, data: dict, evidence: list[EvidenceItem]):
        return ToolObservation(tool_name=tool, status=ObservationStatus.OK, data=data, evidence=evidence)

    def execute(self, claim_id: str, step: AgentPlanStep) -> ToolObservation:
        """
        Only declared methods are dispatched. The planner cannot select a repository method, SQL,
        filesystem path, or a different claim ID.
        """
        tool = step.tool_name
        if tool is None:
            raise ValueError("FINISH is a planner action, not an executable tool")
        arguments = step.validated_arguments()
        if tool == AgentToolName.GET_CLAIM:
            return self._claim_observation(claim_id, tool)
        if tool == AgentToolName.GET_VISUAL_ASSESSMENT:
            return self._visual_observation(claim_id, tool)
        if tool == AgentToolName.GET_HISTORY:
            return self._history_observation(claim_id, tool)
        if tool in ASSESSMENT_TOOLS:
            return self._assessment_observation(claim_id, tool)
        if tool == AgentToolName.FIND_SIMILAR_CLAIMS:
            assert isinstance(arguments, SimilarClaimsInput)
            return self._similar(claim_id, tool, arguments)
        if tool in {AgentToolName.SEARCH_POLICY, AgentToolName.SEARCH_PROCEDURE}:
            assert isinstance(arguments, KnowledgeSearchInput)
            return self._knowledge(claim_id, tool, arguments)
        raise ValueError("Unsupported agent tool")
