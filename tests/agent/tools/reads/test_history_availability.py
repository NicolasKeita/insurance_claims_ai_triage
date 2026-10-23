"""History outages preserve uncertainty and release read sessions."""

import pytest
from sqlalchemy.exc import InterfaceError, OperationalError, ProgrammingError, TimeoutError

import agent.tools._stored as stored_adapters
from agent.models import ObservationStatus
from agent.tools import ReadOnlyAgentTools
from tests.agent.tools._evidence import plan
from tests.agent.tools._repositories import configured as configured


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
