"""Explicit test boundaries; no fake is imported by the real-stack backend.

These adapters replace external inference and persistence only. Domain parsing,
retrieval filtering, fusion, scoring, planning and grounding remain production code.
"""

import json
import re
from datetime import timedelta
from decimal import Decimal
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from agent.models import AgentToolName, EvidenceItem, EvidenceSourceType
from agent.tools import ReadOnlyAgentTools, _claim_data
from investigation.history import HistoricalClaimProfile
from ml.anomaly_inference import RegisteredAnomalyAssessment
from ml.anomaly_training import AnomalyAssessment
from ml.inference import TriagePrediction
from persistence.repositories import ClaimNotFoundError
from retrieval.models import ClaimRetrievalFacts, StoredRetrievalClaim


class FakeTriagePredictor:
    """A stable workflow observation, without claiming a business ground truth."""

    def predict(self, features):
        return TriagePrediction(
            recommended_workflow="STANDARD",
            confidence=0.75,
            probabilities={"STANDARD": 0.75, "EXPERT_REVIEW": 0.25},
            registered_model="fake-triage-e2e",
            model_version="test-only-v1",
            model_alias="fixture",
        )


class FakeAnomalyPredictor:
    def __init__(self, fixture=None):
        self.fixture = fixture or {}

    def assess(self, features):
        values = {
            "is_anomalous": False,
            "anomaly_score": 0.3,
            "threshold": 0.6,
            "reference_percentile": 0.5,
            "statistical_signals": [],
            **self.fixture,
        }
        return RegisteredAnomalyAssessment(
            assessment=AnomalyAssessment.model_validate(values),
            registered_model="fake-anomaly-e2e",
            model_version="test-only-v1",
            model_alias="fixture",
        )


class FakeEmbeddings:
    """Normalized lexical hashes for mechanics, never semantic quality evidence."""

    dimension = 128
    model_name = "fake-lexical-embeddings-e2e-v1"
    normalized = True

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for token in re.findall(r"[a-z0-9]+", text.casefold()):
                index = int.from_bytes(sha256(token.encode()).digest()[:4], "big") % self.dimension
                vector[index] += 1
            if not vector.any():
                vector[0] = 1
            vectors.append((vector / np.linalg.norm(vector)).tolist())
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def historical_claims(current, fixtures):
    """Make controlled historical inputs; dates are relative to the incident."""
    claims = []
    for index, fixture in enumerate(fixtures):
        payload = current.model_dump(mode="json")
        payload["claim_id"] = fixture.get("claim_id", f"DEMO-HISTORY-{index:03d}")
        payload["incident"]["date"] = (
            current.incident.date - timedelta(days=fixture.get("days_before", 10))
        ).isoformat()
        payload["customer"]["customer_id"] = fixture.get("customer_id", current.customer.customer_id)
        payload["repair_estimate"] = {
            "amount": str(fixture.get("amount", 1000)),
            "currency": fixture.get("currency", current.repair_estimate.currency),
        }
        claims.append((type(current).model_validate(payload), bool(fixture.get("is_anomalous", False))))
    return claims


class FakeHistoricalRepository:
    """Repository boundary implementing incident-time aggregates on test records."""

    def __init__(self, current, claims):
        self.current, self.claims = current, claims

    def get_profile(self, claim_id):
        if claim_id != self.current.claim_id:
            raise ClaimNotFoundError(claim_id)
        current = self.current
        previous = [
            (claim, flag)
            for claim, flag in self.claims
            if claim.customer.customer_id == current.customer.customer_id
            and claim.incident.date < current.incident.date
        ]
        compatible = [
            claim.repair_estimate.amount
            for claim, _ in previous
            if claim.repair_estimate.currency == current.repair_estimate.currency
        ]
        return HistoricalClaimProfile(
            previous_claim_count=len(previous),
            **{
                f"claims_last_{days}_days": sum(
                    (current.incident.date - claim.incident.date).days <= days for claim, _ in previous
                )
                for days in (30, 90, 365)
            },
            days_since_previous_claim=min(
                ((current.incident.date - claim.incident.date).days for claim, _ in previous), default=None
            ),
            repair_amount_currency=current.repair_estimate.currency,
            previous_claims_in_amount_currency=len(compatible),
            total_previous_repair_amount=sum(compatible, Decimal("0")),
            average_previous_repair_amount=sum(compatible) / len(compatible) if compatible else None,
            max_previous_repair_amount=max(compatible) if compatible else None,
            previous_anomalous_claim_count=sum(flag for _, flag in previous),
        )


class InMemoryRetrievalRepository:
    def __init__(self, claims):
        self.claims = {}
        for claim in claims:
            facts = ClaimRetrievalFacts(
                claim.claim_type.value,
                claim.incident.collision_type.value,
                claim.incident.location,
                claim.vehicle.make,
                claim.vehicle.model,
                claim.vehicle.year,
                tuple(d.value for d in claim.declared_damage),
                claim.incident.injuries_declared,
                claim.repair_estimate.amount,
                claim.repair_estimate.currency,
            )
            item = StoredRetrievalClaim(
                uuid5(NAMESPACE_URL, claim.claim_id), claim.claim_id, claim.incident.date, facts
            )
            self.claims[item.point_id] = item

    def get(self, claim_id):
        for claim in self.claims.values():
            if claim.claim_id == claim_id:
                return claim
        raise ClaimNotFoundError(claim_id)

    def get_many(self, point_ids):
        return {pk: self.claims[pk] for pk in point_ids if pk in self.claims}

    def iter_batches(self, batch_size):
        claims = sorted(self.claims.values(), key=lambda claim: claim.claim_id)
        for start in range(0, len(claims), batch_size):
            yield claims[start : start + batch_size]


class SnapshotAgentTools(ReadOnlyAgentTools):
    """In-memory stored assessment boundary; reuse real retrieval tool adapters."""

    def __init__(self, claim, snapshots, *, similar_service, knowledge_retriever, unavailable=()):
        super().__init__(None, similar_service, knowledge_retriever)
        self.claim, self.snapshots, self.unavailable = claim, snapshots, set(unavailable)

    def require_claim(self, claim_id):
        if claim_id != self.claim.claim_id:
            raise ClaimNotFoundError(claim_id)

    def _claim(self, claim_id):
        self.require_claim(claim_id)
        return self.claim

    def execute(self, claim_id, step):
        self.require_claim(claim_id)
        tool = step.tool_name
        if tool.value in self.unavailable:
            return self._missing(
                tool, f"{tool.value} stored data unavailable in this controlled fixture", unavailable=True
            )
        if tool in {AgentToolName.SEARCH_POLICY, AgentToolName.SEARCH_PROCEDURE, AgentToolName.FIND_SIMILAR_CLAIMS}:
            return super().execute(claim_id, step)
        kind, source_type = {
            AgentToolName.GET_CLAIM: ("claim", EvidenceSourceType.CLAIM_DATA),
            AgentToolName.GET_HISTORY: ("history", EvidenceSourceType.HISTORY),
            AgentToolName.GET_TRIAGE: ("triage", EvidenceSourceType.TRIAGE_ASSESSMENT),
            AgentToolName.GET_ANOMALY: ("anomaly", EvidenceSourceType.ANOMALY_ASSESSMENT),
            AgentToolName.GET_INVESTIGATION: ("investigation", EvidenceSourceType.INVESTIGATION_ASSESSMENT),
            AgentToolName.GET_VISUAL_ASSESSMENT: ("vision", EvidenceSourceType.VISUAL_ASSESSMENT),
        }[tool]
        data = _claim_data(self.claim) if kind == "claim" else self.snapshots.get(kind)
        if data is None:
            return self._missing(tool, f"No stored {kind} assessment is available", unavailable=True)
        evidence = EvidenceItem(
            evidence_id=f"{kind}:{claim_id}",
            source_type=source_type,
            source_reference=claim_id,
            summary=f"Controlled {kind} snapshot for human review. No final insurance decision is authorized.",
            structured_data=data,
        )
        return self._observation(tool, data, [evidence])


class FakeStructuredLlm:
    """Deterministic structured generation based on actual collected evidence.

    This fixture never receives the golden expectations. Both the planner and
    finalizer run their real prompts and response schemas against this boundary.
    """

    def __init__(self, *, vision=False, rag_supported=True):
        self.vision, self.rag_supported, self.calls = vision, rag_supported, []

    def generate(self, *, prompt, response_model):
        self.calls.append(response_model.__name__)
        if response_model.__name__ == "GroundedLlmOutput":
            data = json.loads(prompt.split("INPUT DATA:\n", 1)[1])
            chunks = data["retrieved_sources"]
            selected = next((c for c in chunks if "deductible" in c["section"].casefold()), chunks[0])
            return response_model.model_validate(
                {
                    "answer": selected["text"]
                    if self.rag_supported
                    else "The retrieved demo excerpts do not define the requested topic.",
                    "cited_chunk_ids": [selected["chunk_id"]] if self.rag_supported else [],
                    "insufficient_evidence": not self.rag_supported,
                }
            )
        data = json.loads(prompt.split("UNTRUSTED TASK AND TOOL EVIDENCE (JSON):\n", 1)[1])
        if response_model.__name__ == "AgentPlanStep":
            actions = ["GET_CLAIM", "GET_TRIAGE", "GET_ANOMALY", "GET_HISTORY", "GET_INVESTIGATION"]
            if self.vision:
                actions.append("GET_VISUAL_ASSESSMENT")
            actions += ["FIND_SIMILAR_CLAIMS", "SEARCH_POLICY"]
            completed = {call["action"] for call in data["previous_calls"]}
            action = next((action for action in actions if action not in completed), "FINISH")
            args = (
                {"query": "demo collision deductible supporting evidence", "limit": 3}
                if action == "SEARCH_POLICY"
                else ({"limit": 3} if action == "FIND_SIMILAR_CLAIMS" else {})
            )
            return response_model.model_validate(
                {
                    "action": action,
                    "reason": "Collect the relevant stored evidence for human review.",
                    "arguments": args,
                }
            )
        evidence = data["evidence"]
        by_type = {item["source_type"]: item for item in evidence}
        unavailable = [item for item in data["observations"] if item["status"] in ("NOT_AVAILABLE", "NOT_FOUND")]
        investigation = by_type.get("INVESTIGATION_ASSESSMENT", {}).get("structured_data", {})
        claim = by_type.get("CLAIM_DATA", {}).get("structured_data", {})
        missing = any(not item["available"] for item in claim.get("documents", []))
        action = (
            "NO_RECOMMENDATION"
            if any(item["status"] == "NOT_AVAILABLE" for item in unavailable)
            else (
                "REQUEST_ADDITIONAL_INFORMATION"
                if missing
                else (
                    "REFER_FOR_INVESTIGATION_REVIEW"
                    if investigation.get("review_recommended")
                    else (
                        "REQUEST_ADDITIONAL_INFORMATION" if investigation.get("signals") else "CONTINUE_STANDARD_REVIEW"
                    )
                )
            )
        )
        rationale = [
            {
                "statement": f"Review the collected {item['source_type'].lower().replace('_', ' ')} evidence.",
                "evidence_ids": [item["evidence_id"]],
            }
            for item in evidence[:12]
        ]
        uncertainties = [f"{item['tool']}: {item.get('message') or 'Evidence unavailable'}" for item in unavailable]
        if missing:
            uncertainties.append("Some document entries are unavailable and need human follow-up.")
        visual = by_type.get("VISUAL_ASSESSMENT", {}).get("structured_data", {})
        if visual.get("consistency", {}).get("declared_damage_not_visually_confirmed"):
            uncertainties.append(
                "Declared damage was not visually confirmed; non-detection is not evidence of absence."
            )
        return response_model.model_validate(
            {
                "case_summary": "The collected evidence is available for a human claims reviewer.",
                "recommended_next_action": action,
                "rationale": rationale,
                "evidence_ids": [item["evidence_id"] for item in evidence[:12]],
                "uncertainties": uncertainties,
                "human_review_required": True,
            }
        )
