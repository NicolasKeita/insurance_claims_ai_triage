"""Test similar claims."""

from datetime import date

from agent.tools import ReadOnlyAgentTools
from retrieval.models import SimilarClaim, SimilarClaimsResponse
from tests.agent.tools._evidence import plan
from tests.agent.tools._repositories import configured
from tests.agent.tools._repositories import configured as configured


def test_similar_context_is_bounded_prior_facts_without_prior_decisions(configured):
    claim, factory, _, _ = configured

    class Similar:
        def search(self, claim_id, limit):
            assert factory.open_count == 0
            assert claim_id == claim.claim_id and limit == 2
            candidates = [
                SimilarClaim(
                    claim_id=claim_id if i == 0 else f"prior-{i}",
                    similarity_score=0.8,
                    incident_date=date(2025, 1, 1),
                    claim_type="AUTO",
                    collision_type="FRONT",
                    repair_amount="100",
                    repair_currency="EUR",
                )
                for i in range(8)
            ]
            return SimilarClaimsResponse(claim_id=claim_id, representation_version="v1", results=candidates)

    result = ReadOnlyAgentTools(factory, similar_service=Similar()).execute(
        claim.claim_id, plan("FIND_SIMILAR_CLAIMS", limit=2)
    )
    assert len(result.evidence) == len(result.data["results"]) == 2
    assert [e.evidence_id for e in result.evidence] == ["similar:prior-1", "similar:prior-2"]
    assert "decision" not in result.data["results"][0]
    assert "Comparative context only" in result.evidence[0].summary
