"""Support."""

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from agent.config import DEFAULT_OBJECTIVE
from agent.models import AgentRecommendation, AgentRunResult
from persistence.repositories import ClaimNotFoundError
from retrieval.models import SimilarClaimsResponse


def make_run(claim_id="demo", objective=None, *, failed=False):
    now = datetime.now(timezone.utc)
    return AgentRunResult(
        run_id=str(uuid4()),
        claim_id=claim_id,
        objective=objective or DEFAULT_OBJECTIVE,
        status="FAILED" if failed else "COMPLETED",
        iteration_count=0,
        tool_call_count=0,
        recommendation=None
        if failed
        else AgentRecommendation(
            case_summary="Available data is insufficient.",
            recommended_next_action="NO_RECOMMENDATION",
            rationale=[],
            evidence_ids=[],
            uncertainties=["No assessment was available."],
        ),
        created_at=now,
        completed_at=now,
        elapsed_seconds=0.01,
        error="Agent generation failed" if failed else None,
    )


class FakeRunStore:
    def __init__(self):
        self.runs = []

    def save(self, run):
        self.runs.append(run)

    def get_by_run_id(self, run_id):
        return next((run for run in self.runs if run.run_id == run_id), None)

    def list_for_claim(self, claim_id, limit=20):
        return [run for run in reversed(self.runs) if run.claim_id == claim_id][:limit]


class FakeAgent:
    def __init__(self, store):
        self.store, self.calls = store, []

    def review_claim(self, claim_id, objective=None):
        self.calls.append((claim_id, objective))
        run = make_run(claim_id, objective)
        self.store.save(run)
        return run


def fake_claim_loader(factory, claim_id):
    if claim_id == "missing":
        raise ClaimNotFoundError("Claim missing does not exist")
    return SimpleNamespace(claim_id=claim_id)


def retrieval_doubles():
    class Similar:
        closed = False

        def search(self, claim_id, limit):
            return SimilarClaimsResponse(claim_id=claim_id, representation_version="claim_retrieval_v1", results=[])

        def close(self):
            self.closed = True

    class Knowledge:
        closed = False

        def search(self, query, source_type, **kwargs):
            return []

        def close(self):
            self.closed = True

    return Similar, Knowledge


def lifespan_loaders(loaded, passed, similar, knowledge):
    def similar_loader(factory):
        loaded.append("similar")
        return similar

    def knowledge_loader():
        loaded.append("knowledge")
        return knowledge

    def agent_loader(factory, **dependencies):
        passed.append(dependencies)

        class CachedAgent:
            def review_claim(self, claim_id, objective=None):
                assert dependencies["similar_service_loader"]() is similar
                assert dependencies["knowledge_retriever_loader"]() is knowledge
                return make_run(claim_id, objective)

        return CachedAgent()

    return similar_loader, knowledge_loader, agent_loader
