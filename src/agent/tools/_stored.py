from agent import tools as adapters
from agent.models import (
    EvidenceItem,
    EvidenceSourceType,
)

from ._data import MAX_SIGNALS, _bounded, _claim_data


def _read_claim(self, claim_id, tool):
    claim = self._claim(claim_id)
    data = _claim_data(claim)
    evidence = EvidenceItem(
        evidence_id=f"claim:{claim_id}",
        source_type=EvidenceSourceType.CLAIM_DATA,
        source_reference=claim_id,
        summary=(
            f"Current claim: {claim.claim_type.value}, incident {claim.incident.date}; "
            f"repair estimate {claim.repair_estimate.amount} {claim.repair_estimate.currency}; "
            f"{len(claim.documents)} document entries and {len(claim.images)} images."
        ),
        structured_data=data,
    )
    return self._observation(tool, data, [evidence])


def _read_history(self, claim_id, tool):
    with self.factory() as session:
        profile = adapters.HistoricalClaimRepository(session).get_profile(claim_id)
    data = profile.model_dump(mode="json")
    evidence = EvidenceItem(
        evidence_id=f"history:{claim_id}",
        source_type=EvidenceSourceType.HISTORY,
        source_reference=claim_id,
        summary=(
            f"Incident-time historical aggregate: {profile.previous_claim_count} earlier claims; "
            f"{profile.claims_last_30_days} in 30 days, {profile.claims_last_365_days} in 365 days. "
            "Prior anomaly flags are statistical model outputs, not established misconduct."
        ),
        structured_data=data,
    )
    return self._observation(tool, data, [evidence])


def _read_assessment(self, claim_id, tool, kind, source_type):
    with self.factory() as session:
        record = getattr(adapters.AssessmentRepository(session), f"get_latest_{kind}")(claim_id)
    if record is None:
        return self._missing(tool, f"No stored {kind} assessment is available", unavailable=True)
    result = record.result.model_dump(mode="json")
    data = {
        "assessment_id": str(record.id),
        "created_at": record.created_at.isoformat(),
        "result": _bounded(result),
    }
    summary = _assessment_summary(self, kind, record, data)
    evidence = EvidenceItem(
        evidence_id=f"{kind}:{record.id}",
        source_type=source_type,
        source_reference=str(record.id),
        summary=summary,
        structured_data=data,
    )
    return self._observation(tool, data, [evidence])


def _assessment_summary(self, kind, record, data):
    if kind == "investigation":
        data["signal_count"] = len(record.result.signals)
        data["omitted_signal_count"] = max(0, len(record.result.signals) - MAX_SIGNALS)
        summary = (
            f"Stored deterministic review prioritization: priority {record.result.review_priority.value}, "
            f"score {record.result.review_score}; {len(record.result.signals)} signals. "
            "This is a human review recommendation, not a fraud determination."
        )
    elif kind == "anomaly":
        summary = (
            f"Stored statistical atypicality assessment: is_anomalous="
            f"{record.result.assessment.is_anomalous}; "
            f"score {record.result.assessment.anomaly_score}. Atypicality does not establish fraud."
        )
    else:
        summary = (
            f"Stored ML workflow recommendation: {record.result.recommended_workflow.value}; "
            f"confidence {record.result.confidence}. This is workflow guidance for human review."
        )
    return summary
