"""Test availability."""

import pytest

from agent.tools import ReadOnlyAgentTools
from retrieval.models import RetrievalUnavailableError
from tests.agent.tools._evidence import plan
from tests.agent.tools._repositories import configured
from tests.agent.tools._repositories import configured as configured


@pytest.mark.parametrize("tool", ["FIND_SIMILAR_CLAIMS", "SEARCH_POLICY", "SEARCH_PROCEDURE"])
@pytest.mark.parametrize(
    "error,expected",
    [(RetrievalUnavailableError("index down"), "NOT_AVAILABLE"), (RuntimeError("unexpected provider bug"), None)],
)
def test_known_unavailability_and_unexpected_errors_have_distinct_behavior(configured, tool, error, expected):
    claim, factory, _, _ = configured

    class Broken:
        def search(self, *_args, **_kwargs):
            raise error

    tools = ReadOnlyAgentTools(factory, similar_service=Broken(), knowledge_retriever=Broken())
    arguments = {} if tool == "FIND_SIMILAR_CLAIMS" else {"query": "coverage"}
    if expected:
        assert tools.execute(claim.claim_id, plan(tool, **arguments)).status == expected
    else:
        with pytest.raises(RuntimeError, match="unexpected provider bug"):
            tools.execute(claim.claim_id, plan(tool, **arguments))


def test_unconfigured_or_empty_retrieval_returns_explicit_status(configured):
    claim, factory, _, _ = configured
    assert (
        ReadOnlyAgentTools(factory).execute(claim.claim_id, plan("SEARCH_POLICY", query="coverage")).status
        == "NOT_AVAILABLE"
    )

    class EmptyRetriever:
        def search(self, *_args, **_kwargs):
            return []

    assert (
        ReadOnlyAgentTools(factory, knowledge_retriever=EmptyRetriever())
        .execute(claim.claim_id, plan("SEARCH_POLICY", query="coverage"))
        .status
        == "NOT_FOUND"
    )
