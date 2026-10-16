"""Seed controlled demo claims and only actually computed assessments."""

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from claims.case import ClaimCase
from claims.models import Claim
from persistence.repositories import ClaimRepository
from persistence.service import PersistenceService
from persistence.vision_assessments import VisionAssessmentRepository


@dataclass(frozen=True)
class DemoAssessments:
    """Pre-computed outputs that may be persisted with a controlled demo claim."""

    triage: Any = None
    anomaly: Any = None
    investigation: Any = None
    vision: Any = None


def seed_case(factory, case: ClaimCase | Claim, assessments: DemoAssessments | None = None):
    """Import a controlled demo claim and append only actually computed outputs."""
    claim = case.claim if isinstance(case, ClaimCase) else case
    with factory() as session:
        existing = ClaimRepository(session).get_by_claim_id(claim.claim_id)
    service = PersistenceService(factory)
    if existing is None:
        service.import_claim(claim)
    elif existing != claim:
        raise ValueError(f"Controlled demo claim ID collision: {claim.claim_id}")
    outputs = assessments or DemoAssessments()
    saved = {}
    if any(value is not None for value in (outputs.triage, outputs.anomaly, outputs.investigation)):
        saved = service.save_assessments(
            claim.claim_id, triage=outputs.triage, anomaly=outputs.anomaly, investigation=outputs.investigation
        )
    if outputs.vision is not None:
        if outputs.vision.claim_id != claim.claim_id:
            raise ValueError("Visual assessment belongs to a different controlled claim")
        with factory.begin() as session:
            saved["vision"] = VisionAssessmentRepository(session).add(outputs.vision)
    return saved


def _history_claim(current: Claim, fixture, index: int) -> Claim:
    payload = current.model_dump(mode="json")
    payload["claim_id"] = fixture.get("claim_id", f"{current.claim_id}-HISTORY-{index}")
    payload["incident"]["date"] = (
        current.incident.date - timedelta(days=fixture.get("days_before", 10))
    ).isoformat()
    payload["customer"]["customer_id"] = fixture.get("customer_id", current.customer.customer_id)
    payload["repair_estimate"] = {
        "amount": str(fixture.get("amount", 1000)),
        "currency": fixture.get("currency", current.repair_estimate.currency),
    }
    return Claim.model_validate(payload)


def seed_history(factory, current: ClaimCase | Claim, fixtures):
    """Seed actual demo claim rows, including same-day/future/customer controls.

    Historical statistical labels are not fabricated. The current benchmark's
    fixtures request no prior anomalies, so missing assessments correctly yield
    zero previous anomaly flags in the real repository's aggregate.
    """
    current = current.claim if isinstance(current, ClaimCase) else current
    if any(fixture.get("is_anomalous", False) for fixture in fixtures):
        raise ValueError(
            "Real evaluation cannot seed fake historical anomaly outputs; supply real assessments separately"
        )
    claims = []
    for index, fixture in enumerate(fixtures):
        claim = _history_claim(current, fixture, index)
        seed_case(factory, claim)
        claims.append(claim)
    return claims