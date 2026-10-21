"""Test validation."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from persistence.agent_runs import AgentRunRepository, agent_run_to_row
from persistence.models import AgentRunRow
from tests.agent.persistence._support import make_agent_run


def test_agent_mapper_revalidates_unsafe_model_copy():
    source = make_agent_run().model_copy(update={"recommendation": None})
    with pytest.raises(ValidationError, match="completed run"):
        agent_run_to_row(uuid4(), source)


def test_agent_mapper_rejects_nonfinite_json_evidence():
    source = make_agent_run()
    source.evidence[0].structured_data["score"] = float("nan")
    with pytest.raises(ValueError, match="JSON compliant"):
        agent_run_to_row(uuid4(), source)


@pytest.mark.parametrize("limit", [0, -1, 101, True, 2.5, "20"])
def test_agent_run_list_limit_is_bounded_before_database_access(limit):
    repository = AgentRunRepository(None)
    with pytest.raises(ValueError, match="between 1 and 100"):
        repository.list_for_claim("claim", limit)


def test_agent_audit_row_has_restricting_fk_and_no_update_delete_api():
    assert next(iter(AgentRunRow.__table__.c.claim_pk.foreign_keys)).ondelete is None
    assert not hasattr(AgentRunRepository, "update")
    assert not hasattr(AgentRunRepository, "delete")
    assert AgentRunRow.__table__.c.insertion_order.identity.always is True
