"""Test snapshots."""

import json
from uuid import uuid4

import pytest

from persistence.agent_runs import (
    agent_run_from_row,
    agent_run_to_row,
)
from tests.agent.persistence._support import add_policy_evidence, make_agent_run


def test_historical_versions_remain_readable_after_current_version_changes():
    run = make_agent_run()
    run.agent_version = run.recommendation.agent_version = "claims_agent_prior_version"
    run.prompt_version = run.recommendation.prompt_version = "claims_agent_prompt_prior_version"
    row = agent_run_to_row(uuid4(), run)
    assert agent_run_from_row(row, run.claim_id) == run


@pytest.mark.parametrize("failed", [False, True])
def test_agent_run_mapping_round_trip_preserves_audit_and_terminal_status(failed):
    source = make_agent_run(failed=failed)
    pk = uuid4()
    row = agent_run_to_row(pk, source)
    assert row.claim_pk == pk
    assert row.run_id == source.run_id
    assert row.status == ("FAILED" if failed else "COMPLETED")
    assert row.recommendation is None if failed else row.recommendation["human_review_required"] is True
    assert agent_run_from_row(row, source.claim_id) == source
    json.dumps([row.tools_used, row.evidence, row.trace, row.recommendation], allow_nan=False)


def test_agent_run_mapping_detaches_json_snapshots():
    source = make_agent_run()
    row = agent_run_to_row(uuid4(), source)
    source.evidence[0].structured_data["declared_damage"].append("REAR_BUMPER")
    source.trace[0].tool_args["changed_later"] = True
    source.recommendation.uncertainties.append("Added after snapshot.")
    assert row.evidence[0]["structured_data"]["declared_damage"] == ["FRONT_BUMPER"]
    assert row.trace[0]["tool_args"] == {}
    assert row.recommendation["uncertainties"] == ["Policy evidence has not been collected."]


def test_agent_audit_mapping_preserves_policy_citation_metadata():
    source = add_policy_evidence(make_agent_run())
    row = agent_run_to_row(uuid4(), source)
    loaded = agent_run_from_row(row, source.claim_id)
    assert loaded == source
    citation = loaded.recommendation.citations[0]
    assert (citation.chunk_id, citation.source_id, citation.source_version, citation.section) == (
        "demo-policy-v1-collision-0",
        "demo-auto-policy",
        "v1",
        "Collision review",
    )
    assert loaded.evidence[-1].structured_data == citation.model_dump(mode="json")
