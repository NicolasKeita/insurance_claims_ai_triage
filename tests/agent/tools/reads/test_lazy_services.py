"""Test lazy services."""

from agent.tools import ReadOnlyAgentTools
from retrieval.knowledge.models import KnowledgeFilter
from tests.agent.tools._evidence import chunk, plan
from tests.agent.tools._repositories import configured
from tests.agent.tools._repositories import configured as configured


def test_retrieval_callbacks_load_once_and_sessions_close(configured):
    claim, factory, _, _ = configured
    loaded, searched, closed = [], [], []

    class Retriever:
        def search(self, query, source_type, **filters):
            assert factory.open_count == 0
            searched.append((query, source_type, filters))
            return [chunk(claim)]

        def close(self):
            closed.append(True)

    def loader():
        assert factory.open_count == 0
        loaded.append(True)
        return Retriever()

    tools = ReadOnlyAgentTools(factory, knowledge_retriever_loader=loader)
    for _ in range(2):
        result = tools.execute(claim.claim_id, plan("SEARCH_POLICY", query="collision coverage", limit=2))
        assert result.evidence[0].evidence_id == "knowledge:synthetic-policy-chunk"
    assert loaded == [True]
    expected_filters = KnowledgeFilter(product=claim.policy.product, as_of_date=claim.incident.date)
    assert all(filters == {"filters": expected_filters, "limit": 2} for _, _, filters in searched)
    tools.close()
    assert closed == [True]
