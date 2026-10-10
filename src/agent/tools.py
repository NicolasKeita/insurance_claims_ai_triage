"""Bounded, claim-scoped read adapters around the existing repositories/services.

These adapters never compute or persist a business assessment. New SQL sessions
close before embedding/retrieval calls; unavailable assessments stay unavailable.
"""

from collections import Counter
from collections.abc import Callable
from typing import Protocol

from agent.models import (
    AgentPlanStep, AgentToolName, EvidenceItem, EvidenceSourceType,
    KnowledgeSearchInput, ObservationStatus, SimilarClaimsInput, ToolObservation,
)
from claims.models import Claim
from knowledge.models import KnowledgeSourceType, applicable_on
from persistence.history import HistoricalClaimRepository
from persistence.repositories import AssessmentRepository, ClaimNotFoundError, ClaimRepository
from retrieval.models import RetrievalUnavailableError

MAX_SIGNALS = 12
MAX_VALUE_CHARS = 800
MAX_KNOWLEDGE_EXCERPT_CHARS = 3200


class AgentTools(Protocol):
    def require_claim(self, claim_id: str) -> None: ...
    def execute(self, claim_id: str, step: AgentPlanStep) -> ToolObservation: ...


def _bounded(value, depth: int = 0, budget: list[int] | None = None):
    """Bound persisted free text, nested signal facts, and collection fan-out."""
    budget = [12000] if budget is None else budget
    if budget[0] <= 0:
        return "[data budget exhausted]"
    if isinstance(value, str):
        result = value[:min(MAX_VALUE_CHARS, budget[0])]
        budget[0] -= len(result)
        return result
    if isinstance(value, (bool, int, float)) or value is None:
        budget[0] -= len(str(value))
        return value
    if depth >= 4:
        return "[nested data omitted]"
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:20]:
            if budget[0] <= 0:
                break
            key = str(key)[:100]
            budget[0] -= len(key)
            result[key] = _bounded(item, depth + 1, budget)
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value[:MAX_SIGNALS]:
            if budget[0] <= 0:
                break
            result.append(_bounded(item, depth + 1, budget))
        return result
    # Repository model_dump(mode="json") produces JSON values; unexpected
    # objects fail explicitly rather than hiding infrastructure/schema changes.
    raise TypeError("Agent tool data must be JSON-compatible")


def _claim_data(claim: Claim) -> dict:
    documents = Counter((d.type.value, d.available) for d in claim.documents)
    return {
        "claim_id": claim.claim_id,
        "claim_type": claim.claim_type.value,
        "status": claim.status.value,
        "policy_product": claim.policy.product[:MAX_VALUE_CHARS],
        "incident": _bounded(claim.incident.model_dump(mode="json")),
        "vehicle": _bounded(claim.vehicle.model_dump(mode="json")),
        "repair_estimate": claim.repair_estimate.model_dump(mode="json"),
        "declared_damage": sorted({damage.value for damage in claim.declared_damage}),
        "documents": [{"type": kind, "available": available, "count": count}
                      for (kind, available), count in sorted(documents.items())],
        "document_count": len(claim.documents),
        "image_count": len(claim.images),
    }


class ReadOnlyAgentTools:
    def __init__(
        self, factory, similar_service=None, knowledge_retriever=None, *,
        similar_service_loader: Callable | None = None,
        knowledge_retriever_loader: Callable | None = None,
    ):
        self.factory = factory
        self.similar_service = similar_service
        self.knowledge_retriever = knowledge_retriever
        self.similar_service_loader = similar_service_loader
        self.knowledge_retriever_loader = knowledge_retriever_loader

    def close(self) -> None:
        """Release standalone retrieval resources; API lifespan owns its caches."""
        seen = set()
        for service in (self.similar_service, self.knowledge_retriever):
            if service is not None and id(service) not in seen:
                seen.add(id(service))
                if hasattr(service, "close"):
                    service.close()

    def require_claim(self, claim_id: str) -> None:
        with self.factory() as session:
            ClaimRepository(session).require_pk(claim_id)

    def _claim(self, claim_id: str) -> Claim:
        with self.factory() as session:
            claim = ClaimRepository(session).get_by_claim_id(claim_id)
            if claim is None:
                raise ClaimNotFoundError(f"Claim {claim_id} does not exist")
            return claim

    @staticmethod
    def _missing(tool: AgentToolName, message: str, *, unavailable=False):
        return ToolObservation(
            tool_name=tool,
            status=ObservationStatus.NOT_AVAILABLE if unavailable else ObservationStatus.NOT_FOUND,
            data=None, evidence=[], message=message,
        )

    @staticmethod
    def _observation(tool: AgentToolName, data: dict, evidence: list[EvidenceItem]):
        return ToolObservation(tool_name=tool, status=ObservationStatus.OK,
                               data=data, evidence=evidence)

    def execute(self, claim_id: str, step: AgentPlanStep) -> ToolObservation:
        tool = step.tool_name
        if tool is None:
            raise ValueError("FINISH is a planner action, not an executable tool")
        arguments = step.validated_arguments()
        # Only declared methods are dispatched. The planner cannot select a
        # repository method, SQL, filesystem path, or a different claim ID.
        if tool == AgentToolName.GET_CLAIM:
            claim = self._claim(claim_id)
            data = _claim_data(claim)
            evidence = EvidenceItem(
                evidence_id=f"claim:{claim_id}", source_type=EvidenceSourceType.CLAIM_DATA,
                source_reference=claim_id,
                summary=(f"Current claim: {claim.claim_type.value}, incident {claim.incident.date}; "
                         f"repair estimate {claim.repair_estimate.amount} {claim.repair_estimate.currency}; "
                         f"{len(claim.documents)} document entries and {len(claim.images)} images."),
                structured_data=data,
            )
            return self._observation(tool, data, [evidence])
        if tool == AgentToolName.GET_HISTORY:
            with self.factory() as session:
                profile = HistoricalClaimRepository(session).get_profile(claim_id)
            data = profile.model_dump(mode="json")
            evidence = EvidenceItem(
                evidence_id=f"history:{claim_id}", source_type=EvidenceSourceType.HISTORY,
                source_reference=claim_id,
                summary=(f"Incident-time historical aggregate: {profile.previous_claim_count} earlier claims; "
                         f"{profile.claims_last_30_days} in 30 days, {profile.claims_last_365_days} in 365 days. "
                         "Prior anomaly flags are statistical model outputs, not established misconduct."),
                structured_data=data,
            )
            return self._observation(tool, data, [evidence])
        assessments = {
            AgentToolName.GET_TRIAGE: ("triage", EvidenceSourceType.TRIAGE_ASSESSMENT),
            AgentToolName.GET_ANOMALY: ("anomaly", EvidenceSourceType.ANOMALY_ASSESSMENT),
            AgentToolName.GET_INVESTIGATION: ("investigation", EvidenceSourceType.INVESTIGATION_ASSESSMENT),
        }
        if tool in assessments:
            kind, source_type = assessments[tool]
            with self.factory() as session:
                record = getattr(AssessmentRepository(session), f"get_latest_{kind}")(claim_id)
            if record is None:
                return self._missing(tool, f"No stored {kind} assessment is available", unavailable=True)
            result = record.result.model_dump(mode="json")
            data = {"assessment_id": str(record.id), "created_at": record.created_at.isoformat(),
                    "result": _bounded(result)}
            if kind == "investigation":
                data["signal_count"] = len(record.result.signals)
                data["omitted_signal_count"] = max(0, len(record.result.signals) - MAX_SIGNALS)
                summary = (f"Stored deterministic review prioritization: priority {record.result.review_priority.value}, "
                           f"score {record.result.review_score}; {len(record.result.signals)} signals. "
                           "This is a human review recommendation, not a fraud determination.")
            elif kind == "anomaly":
                summary = (f"Stored statistical atypicality assessment: is_anomalous="
                           f"{record.result.assessment.is_anomalous}; "
                           f"score {record.result.assessment.anomaly_score}. Atypicality does not establish fraud.")
            else:
                summary = (f"Stored ML workflow recommendation: {record.result.recommended_workflow.value}; "
                           f"confidence {record.result.confidence}. This is workflow guidance for human review.")
            evidence = EvidenceItem(
                evidence_id=f"{kind}:{record.id}", source_type=source_type,
                source_reference=str(record.id), summary=summary, structured_data=data,
            )
            return self._observation(tool, data, [evidence])
        if tool == AgentToolName.FIND_SIMILAR_CLAIMS:
            assert isinstance(arguments, SimilarClaimsInput)
            return self._similar(claim_id, tool, arguments)
        if tool in {AgentToolName.SEARCH_POLICY, AgentToolName.SEARCH_PROCEDURE}:
            assert isinstance(arguments, KnowledgeSearchInput)
            return self._knowledge(claim_id, tool, arguments)
        raise ValueError("Unsupported agent tool")

    def _similar(self, claim_id, tool, arguments):
        claim = self._claim(claim_id)  # Session closes before external service calls.
        try:
            service = self.similar_service
            if service is None and self.similar_service_loader is not None:
                service = self.similar_service_loader()
                self.similar_service = service
            if service is None:
                return self._missing(tool, "Similar claims retrieval is not configured", unavailable=True)
            response = service.search(claim_id, limit=arguments.limit)
        except RetrievalUnavailableError:
            return self._missing(tool, "Similar claims retrieval is unavailable", unavailable=True)
        if response.claim_id != claim_id:
            raise ValueError("Similar claims response violates current claim scope")
        evidence, results, seen = [], [], set()
        for candidate in response.results:
            if (candidate.claim_id == claim_id or candidate.claim_id in seen
                    or candidate.incident_date >= claim.incident.date):
                continue
            seen.add(candidate.claim_id)
            data = _bounded(candidate.model_dump(mode="json"))
            results.append(data)
            evidence.append(EvidenceItem(
                evidence_id=f"similar:{candidate.claim_id}", source_type=EvidenceSourceType.SIMILAR_CLAIM,
                source_reference=candidate.claim_id,
                summary=(f"Comparative context only: earlier {candidate.claim_type[:100]} claim dated "
                         f"{candidate.incident_date}, collision {candidate.collision_type[:100]}, "
                         f"repair {candidate.repair_amount} {candidate.repair_currency[:10]}. "
                         "Similarity is not evidence of misconduct or a basis for copying a prior decision."),
                structured_data=data,
            ))
            if len(results) == arguments.limit:
                break
        if not evidence:
            return self._missing(tool, "No eligible similar claims were retrieved")
        return self._observation(tool, {"results": results,
                                       "representation_version": response.representation_version[:100]}, evidence)

    def _knowledge(self, claim_id, tool, arguments):
        claim = self._claim(claim_id)  # Authoritative product/date cannot be set by the planner.
        policy = tool == AgentToolName.SEARCH_POLICY
        source_type = KnowledgeSourceType.POLICY if policy else KnowledgeSourceType.PROCEDURE
        product = claim.policy.product if policy else None
        as_of_date = claim.incident.date if policy else None
        try:
            retriever = self.knowledge_retriever
            if retriever is None and self.knowledge_retriever_loader is not None:
                retriever = self.knowledge_retriever_loader()
                self.knowledge_retriever = retriever
            if retriever is None:
                return self._missing(tool, "Knowledge retrieval is not configured", unavailable=True)
            chunks = retriever.search(arguments.query, source_type, product=product,
                                      as_of_date=as_of_date, limit=arguments.limit)
        except RetrievalUnavailableError:
            return self._missing(tool, "Knowledge retrieval is unavailable", unavailable=True)
        evidence, summaries, seen = [], [], set()
        for chunk in chunks:
            if (chunk.source_type != source_type or chunk.chunk_id in seen
                    or (policy and chunk.product != product) or not applicable_on(chunk, as_of_date)):
                continue
            seen.add(chunk.chunk_id)
            # Preserve exact authoritative citation fields within a finite
            # context; oversized metadata is a source-contract failure.
            metadata_limits = {"chunk_id": 250, "source_id": 300, "source_version": 100,
                               "title": 500, "section": 500, "language": 16}
            if any(len(getattr(chunk, field)) > maximum
                   for field, maximum in metadata_limits.items()):
                raise ValueError("Knowledge citation metadata exceeds agent context limits")
            metadata = {
                "chunk_id": chunk.chunk_id, "source_id": chunk.source_id,
                "source_version": chunk.source_version, "title": chunk.title,
                "section": chunk.section, "product": chunk.product[:MAX_VALUE_CHARS] if chunk.product else None,
                "language": chunk.language,
                "effective_from": str(chunk.effective_from) if chunk.effective_from else None,
                "effective_to": str(chunk.effective_to) if chunk.effective_to else None,
                "excerpt_truncated": len(chunk.text) > MAX_KNOWLEDGE_EXCERPT_CHARS,
            }
            evidence.append(EvidenceItem(
                evidence_id=f"knowledge:{chunk.chunk_id}",
                source_type=EvidenceSourceType.POLICY_CHUNK if policy else EvidenceSourceType.PROCEDURE_CHUNK,
                source_reference=chunk.chunk_id,
                summary=chunk.text[:MAX_KNOWLEDGE_EXCERPT_CHARS], structured_data=metadata,
            ))
            # Ranking depends on the query; keep it in the observation rather
            # than changing a stable chunk's evidence payload between searches.
            summaries.append({**metadata, "similarity_score": chunk.similarity_score})
            if len(evidence) == arguments.limit:
                break
        if not evidence:
            return self._missing(tool, "No applicable knowledge excerpts were retrieved")
        return self._observation(tool, {"source_type": source_type.value,
                                       "results": summaries}, evidence)
