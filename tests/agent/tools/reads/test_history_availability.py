"""History outages preserve uncertainty and release read sessions."""

import pytest
from sqlalchemy.exc import InterfaceError, OperationalError, ProgrammingError, TimeoutError

import agent.tools._stored as stored_adapters
from agent.models import ObservationStatus
from agent.tools import ReadOnlyAgentTools
from tests.agent.tools._evidence import plan
from tests.agent.tools._repositories import configured as configured


def test_history_uses_the_regular_session_factory_by_default(configured):
    claim, factory, calls, _ = configured
    tools = ReadOnlyAgentTools(factory)

    observation = tools.execute(claim.claim_id, plan("GET_HISTORY"))

    assert tools.history_session_factory is factory
    assert observation.status == ObservationStatus.OK
    assert calls.count(("history", claim.claim_id)) == 1
    assert factory.open_count == 0


def test_scoped_history_outage_keeps_other_repository_reads_available(configured):
    claim, factory, calls, _ = configured

    class UnavailableHistorySessions:
        attempts = 0

        def __bool__(self):
            return False

        def __call__(self):
            self.attempts += 1
            raise OperationalError(None, None, RuntimeError("controlled history connection failure"))

    unavailable = UnavailableHistorySessions()
    tools = ReadOnlyAgentTools(factory, history_session_factory=unavailable)
    assert tools.history_session_factory is unavailable
    tools.require_claim(claim.claim_id)
    for action in ("GET_CLAIM", "GET_TRIAGE", "GET_ANOMALY", "GET_INVESTIGATION"):
        assert tools.execute(claim.claim_id, plan(action)).status == ObservationStatus.OK
    assert unavailable.attempts == 0

    observation = tools.execute(claim.claim_id, plan("GET_HISTORY"))

    assert unavailable.attempts == 1
    assert observation.status == ObservationStatus.NOT_AVAILABLE
    assert observation.data is None and observation.evidence == []
    assert ("history", claim.claim_id) not in calls
    assert tools.execute(claim.claim_id, plan("GET_CLAIM")).status == ObservationStatus.OK
    assert factory.open_count == 0


@pytest.mark.parametrize(
    "error",
    [
        OperationalError("SELECT private_history", {"customer": "private"}, RuntimeError("secret host")),
        InterfaceError("SELECT private_history", {"customer": "private"}, RuntimeError("secret driver")),
        TimeoutError("private connection pool details"),
    ],
)
def test_operational_history_failures_return_sanitized_unavailability(configured, monkeypatch, error):
    claim, factory, _, _ = configured

    class UnavailableHistory:
        def __init__(self, session):
            assert session.open_count == 1

        def get_profile(self, claim_id):
            raise error

    monkeypatch.setattr(stored_adapters, "HistoricalClaimRepository", UnavailableHistory)

    observation = ReadOnlyAgentTools(factory).execute(claim.claim_id, plan("GET_HISTORY"))

    assert observation.status == ObservationStatus.NOT_AVAILABLE
    assert observation.message == "Historical claim data is temporarily unavailable"
    assert observation.data is None
    assert observation.evidence == []
    assert factory.open_count == 0
    assert "private" not in observation.model_dump_json()
    assert "secret" not in observation.model_dump_json()


@pytest.mark.parametrize(
    "error",
    [
        ProgrammingError("SELECT wrong_column", {}, RuntimeError("query bug")),
        RuntimeError("unexpected history bug"),
    ],
)
def test_history_programming_failures_remain_errors(configured, monkeypatch, error):
    claim, factory, _, _ = configured

    class BrokenHistory:
        def __init__(self, session):
            assert session.open_count == 1

        def get_profile(self, claim_id):
            raise error

    monkeypatch.setattr(stored_adapters, "HistoricalClaimRepository", BrokenHistory)

    with pytest.raises(type(error)) as raised:
        ReadOnlyAgentTools(factory).execute(claim.claim_id, plan("GET_HISTORY"))

    assert raised.value is error
    assert factory.open_count == 0
