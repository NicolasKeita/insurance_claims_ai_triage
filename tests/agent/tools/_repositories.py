"""Repositories."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

import agent.tools._adapter as claim_adapters
import agent.tools._stored as assessment_adapters
from claims.investigation.history import HistoricalClaimProfile
from claims.loader import load_claim_case
from persistence.mappers import AssessmentRecord
from persistence.repositories import ClaimNotFoundError
from tests.agent.tools._constants import ASSESSMENT_ID
from tests.domain.persistence.mapping._support import (
    make_anomaly,
    make_investigation,
    make_triage,
)


class ReadSessionFactory:
    def __init__(self):
        self.open_count = 0

    def __call__(self):
        return self

    def __enter__(self):
        self.open_count += 1
        return self

    def __exit__(self, *_):
        self.open_count -= 1

    def add(self, *_):
        raise AssertionError("Business writes are forbidden")

    flush = commit = add


@pytest.fixture
def configured(monkeypatch):
    claim = load_claim_case(Path("data/CLAIM-2026-00001")).claim
    factory, calls = ReadSessionFactory(), []
    records = {
        kind: AssessmentRecord(ASSESSMENT_ID, datetime(2026, 9, 25, tzinfo=timezone.utc), result)
        for kind, result in {
            "triage": make_triage(),
            "anomaly": make_anomaly(),
            "investigation": make_investigation(),
        }.items()
    }

    Claims = claim_repository(claim, calls)
    History = history_repository(factory, calls, Claims)
    Assessments = assessment_repository(factory, calls, records, Claims)
    monkeypatch.setattr(claim_adapters, "ClaimRepository", Claims)
    monkeypatch.setattr(assessment_adapters, "HistoricalClaimRepository", History)
    monkeypatch.setattr(assessment_adapters, "AssessmentRepository", Assessments)
    return claim, factory, calls, records


def claim_repository(claim, calls):
    class Claims:
        def __init__(self, session):
            assert session.open_count == 1

        def require_pk(self, claim_id):
            calls.append(("require", claim_id))
            if claim_id != claim.claim_id:
                raise ClaimNotFoundError("Claim does not exist")
            return ASSESSMENT_ID

        def get_by_claim_id(self, claim_id):
            calls.append(("claim", claim_id))
            return claim if claim_id == claim.claim_id else None

    return Claims


def history_repository(factory, calls, Claims):
    class History:
        def __init__(self, session):
            assert session.open_count == 1

        def get_profile(self, claim_id):
            calls.append(("history", claim_id))
            Claims(factory).require_pk(claim_id)
            return HistoricalClaimProfile()

    return History


def assessment_repository(factory, calls, records, Claims):
    class Assessments:
        def __init__(self, session):
            assert session.open_count == 1

        def _get(self, claim_id, kind):
            calls.append((kind, claim_id))
            Claims(factory).require_pk(claim_id)
            return records[kind]

        def get_latest_triage(self, claim_id):
            return self._get(claim_id, "triage")

        def get_latest_anomaly(self, claim_id):
            return self._get(claim_id, "anomaly")

        def get_latest_investigation(self, claim_id):
            return self._get(claim_id, "investigation")

        def add_triage(self, *_):
            raise AssertionError("Assessments must never be created")

        add_anomaly = add_investigation = add_triage

    return Assessments
