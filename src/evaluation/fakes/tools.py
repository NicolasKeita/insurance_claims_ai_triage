"""Agent tool boundary backed by controlled in-memory assessments."""

from agent.models import AgentToolName, EvidenceItem, EvidenceSourceType
from agent.tools import ReadOnlyAgentTools, _claim_data
from persistence.repositories import ClaimNotFoundError

TOOL_SOURCES = {
    AgentToolName.GET_CLAIM: ("claim", EvidenceSourceType.CLAIM_DATA),
    AgentToolName.GET_HISTORY: ("history", EvidenceSourceType.HISTORY),
    AgentToolName.GET_TRIAGE: ("triage", EvidenceSourceType.TRIAGE_ASSESSMENT),
    AgentToolName.GET_ANOMALY: ("anomaly", EvidenceSourceType.ANOMALY_ASSESSMENT),
    AgentToolName.GET_INVESTIGATION: ("investigation", EvidenceSourceType.INVESTIGATION_ASSESSMENT),
    AgentToolName.GET_VISUAL_ASSESSMENT: ("vision", EvidenceSourceType.VISUAL_ASSESSMENT),
}
RETRIEVAL_TOOLS = {AgentToolName.SEARCH_POLICY, AgentToolName.SEARCH_PROCEDURE, AgentToolName.FIND_SIMILAR_CLAIMS}


class SnapshotAgentTools(ReadOnlyAgentTools):
    """In-memory stored assessment boundary; reuse real retrieval tool adapters."""

    def __init__(self, claim, snapshots, *, similar_service, knowledge_retriever, unavailable=()):
        super().__init__(None, similar_service, knowledge_retriever)
        self.claim, self.snapshots, self.unavailable = claim, snapshots, set(unavailable)

    def require_claim(self, claim_id):
        if claim_id != self.claim.claim_id:
            raise ClaimNotFoundError(claim_id)

    def _claim(self, claim_id):
        self.require_claim(claim_id)
        return self.claim

    def _stored_evidence(self, claim_id, tool):
        kind, source_type = TOOL_SOURCES[tool]
        data = _claim_data(self.claim) if kind == "claim" else self.snapshots.get(kind)
        if data is None:
            return None
        evidence = EvidenceItem(
            evidence_id=f"{kind}:{claim_id}",
            source_type=source_type,
            source_reference=claim_id,
            summary=f"Controlled {kind} snapshot for human review. No final insurance decision is authorized.",
            structured_data=data,
        )
        return self._observation(tool, data, [evidence])

    def execute(self, claim_id, step):
        self.require_claim(claim_id)
        tool = step.tool_name
        if tool.value in self.unavailable:
            return self._missing(
                tool, f"{tool.value} stored data unavailable in this controlled fixture", unavailable=True
            )
        if tool in RETRIEVAL_TOOLS:
            return super().execute(claim_id, step)
        observation = self._stored_evidence(claim_id, tool)
        if observation is None:
            return self._missing(tool, f"No stored {TOOL_SOURCES[tool][0]} assessment is available", unavailable=True)
        return observation
