"""Stored triage, anomaly and investigation assessment observations."""

from agent.models import AgentToolName, EvidenceItem, EvidenceSourceType
from agent.tools.bounding import MAX_SIGNALS, bounded
from persistence.repositories import AssessmentRepository

ASSESSMENT_TOOLS = {
    AgentToolName.GET_TRIAGE: ("triage", EvidenceSourceType.TRIAGE_ASSESSMENT),
    AgentToolName.GET_ANOMALY: ("anomaly", EvidenceSourceType.ANOMALY_ASSESSMENT),
    AgentToolName.GET_INVESTIGATION: ("investigation", EvidenceSourceType.INVESTIGATION_ASSESSMENT),
}


class AssessmentObservations:
    """Build the stored-assessment tool observations."""

    def _assessment_observation(self, claim_id, tool):
        kind, source_type = ASSESSMENT_TOOLS[tool]
        with self.factory() as session:
            record = getattr(AssessmentRepository(session), f"get_latest_{kind}")(claim_id)
        if record is None:
            return self._missing(tool, f"No stored {kind} assessment is available", unavailable=True)
        result = record.result.model_dump(mode="json")
        data = {
            "assessment_id": str(record.id),
            "created_at": record.created_at.isoformat(),
            "result": bounded(result),
        }
        if kind == "investigation":
            data["signal_count"] = len(record.result.signals)
            data["omitted_signal_count"] = max(0, len(record.result.signals) - MAX_SIGNALS)
        evidence = EvidenceItem(
            evidence_id=f"{kind}:{record.id}",
            source_type=source_type,
            source_reference=str(record.id),
            summary=_assessment_summary(kind, record),
            structured_data=data,
        )
        return self._observation(tool, data, [evidence])


def _assessment_summary(kind, record):
    if kind == "investigation":
        return (
            f"Stored deterministic review prioritization: priority {record.result.review_priority.value}, "
            f"score {record.result.review_score}; {len(record.result.signals)} signals. "
            "This is a human review recommendation, not a fraud determination."
        )
    if kind == "anomaly":
        return (
            f"Stored statistical atypicality assessment: is_anomalous="
            f"{record.result.assessment.is_anomalous}; "
            f"score {record.result.assessment.anomaly_score}. Atypicality does not establish fraud."
        )
    return (
        f"Stored ML workflow recommendation: {record.result.recommended_workflow.value}; "
        f"confidence {record.result.confidence}. This is workflow guidance for human review."
    )
