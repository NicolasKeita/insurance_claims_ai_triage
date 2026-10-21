"""Append-only assessment histories; flushes but never commits."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from claims.investigation.models import InvestigationAssessment
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.inference import TriagePrediction
from persistence.mappers import (
    anomaly_from_row,
    anomaly_to_row,
    investigation_from_row,
    investigation_to_row,
    triage_from_row,
    triage_to_row,
)
from persistence.models import AnomalyAssessmentRow, InvestigationAssessmentRow, TriageAssessmentRow
from persistence.repositories.claims import ClaimRepository


class AssessmentRepository:
    """Append/read API only. Anomaly uses (created_at, insertion_order).

    Other histories retain their existing stable UUID tie-break. Anomaly's
    database counter orders timestamp ties by allocation, not commit time.
    """

    def __init__(self, session: Session):
        self.session = session
        self.claims = ClaimRepository(session)

    def _add(self, claim_id, result, to_row, from_row):
        row = to_row(self.claims.require_pk(claim_id), result)
        self.session.add(row)
        self.session.flush()
        return from_row(row)

    def add_triage(self, claim_id: str, result: TriagePrediction):
        return self._add(claim_id, result, triage_to_row, triage_from_row)

    def add_anomaly(self, claim_id: str, result: RegisteredAnomalyAssessment):
        return self._add(claim_id, result, anomaly_to_row, anomaly_from_row)

    def add_investigation(self, claim_id: str, result: InvestigationAssessment):
        return self._add(claim_id, result, investigation_to_row, investigation_from_row)

    def _read(self, claim_id, model, from_row, latest=False):
        pk = self.claims.require_pk(claim_id)
        query = select(model).where(model.claim_pk == pk)
        tie_break = model.insertion_order if model is AnomalyAssessmentRow else model.id
        if latest:
            row = self.session.scalar(query.order_by(model.created_at.desc(), tie_break.desc()).limit(1))
            return from_row(row) if row else None
        return [from_row(row) for row in self.session.scalars(query.order_by(model.created_at, tie_break))]

    def list_triage_history(self, claim_id: str):
        return self._read(claim_id, TriageAssessmentRow, triage_from_row)

    def list_anomaly_history(self, claim_id: str):
        return self._read(claim_id, AnomalyAssessmentRow, anomaly_from_row)

    def list_investigation_history(self, claim_id: str):
        return self._read(claim_id, InvestigationAssessmentRow, investigation_from_row)

    def get_latest_triage(self, claim_id: str):
        return self._read(claim_id, TriageAssessmentRow, triage_from_row, latest=True)

    def get_latest_anomaly(self, claim_id: str):
        return self._read(claim_id, AnomalyAssessmentRow, anomaly_from_row, latest=True)

    def get_latest_investigation(self, claim_id: str):
        return self._read(claim_id, InvestigationAssessmentRow, investigation_from_row, latest=True)
