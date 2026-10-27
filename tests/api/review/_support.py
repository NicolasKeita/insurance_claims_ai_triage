"""Controlled repositories and stored domain snapshots; no inference."""

from contextlib import nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID

from claims.investigation.history import HistoricalClaimProfile
from claims.review.persisted import PersistedReviewService
from tests.agent.persistence._support import add_policy_evidence, make_agent_run
from tests.domain.investigation.multimodal._assessment import make_visual_assessment
from tests.domain.persistence.mapping._support import make_anomaly, make_investigation, make_triage

TIMESTAMP = datetime(2026, 10, 10, 10, tzinfo=timezone.utc)


def configure_repositories(monkeypatch, claim, *, complete=False, asset_root=None):
    from claims.review import persisted

    fake = RepositoryFake(claim, _records(claim, complete))
    for name in ("ClaimRepository", "AssessmentRepository", "HistoricalClaimRepository"):
        monkeypatch.setattr(persisted, name, lambda _: fake)
    monkeypatch.setattr(persisted, "VisionAssessmentRepository", lambda _: _snapshot(fake, "vision"))
    monkeypatch.setattr(persisted, "AgentRunRepository", lambda _: _snapshot(fake, "agent"))
    return PersistedReviewService(lambda: nullcontext(object()), asset_root=asset_root), fake.records, fake.reads


class RepositoryFake:
    def __init__(self, claim, records):
        self.claim, self.records, self.reads = claim, records, []

    def read(self, kind):
        self.reads.append(kind)
        value = self.records.get(kind)
        if isinstance(value, Exception):
            raise value
        return value

    def get_by_claim_id(self, claim_id):
        self.reads.append("claim")
        return self.claim if claim_id == self.claim.claim_id else None

    def get_profile(self, claim_id):
        return self.read("history")

    def __getattr__(self, name):
        if name.startswith("get_latest_"):
            return lambda claim_id: self.read(name.removeprefix("get_latest_"))
        raise AttributeError(name)


def _snapshot(fake, kind):
    return SimpleNamespace(latest_for_claim=lambda claim_id: fake.read(kind))


def _records(claim, complete):
    results = {"history": HistoricalClaimProfile()}
    if complete:
        results.update(
            {
                "triage": record(make_triage("triage-3")),
                "anomaly": record(make_anomaly("anomaly-2")),
                "investigation": record(make_investigation()),
                "vision": record(make_visual_assessment(claim_id=claim.claim_id)),
                "agent": add_policy_evidence(make_agent_run(claim.claim_id)),
            }
        )
    return results


def record(result):
    return SimpleNamespace(id=UUID("00000000-0000-0000-0000-000000000033"), created_at=TIMESTAMP, result=result)


def forbidden(*args, **kwargs):
    raise AssertionError("Opening a review must never initialize an inference or retrieval service")
