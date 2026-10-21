from agent.models import (
    EvidenceItem,
    EvidenceSourceType,
)
from retrieval.models import RetrievalUnavailableError

from ._data import _bounded


def _similar(self, claim_id, tool, arguments):
    """
    Session closes before external service calls.
    """
    claim = self._claim(claim_id)
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
    evidence, results = _similar_evidence(self, response, claim, arguments)
    if not evidence:
        return self._missing(tool, "No eligible similar claims were retrieved")
    return self._observation(
        tool,
        {
            "results": results,
            "representation_version": response.representation_version[:100],
        },
        evidence,
    )


def _similar_evidence(self, response, claim, arguments):
    evidence, results, seen = [], [], set()
    for candidate in response.results:
        if (
            candidate.claim_id == claim.claim_id
            or candidate.claim_id in seen
            or candidate.incident_date >= claim.incident.date
        ):
            continue
        seen.add(candidate.claim_id)
        data = _bounded(candidate.model_dump(mode="json"))
        results.append(data)
        evidence.append(
            EvidenceItem(
                evidence_id=f"similar:{candidate.claim_id}",
                source_type=EvidenceSourceType.SIMILAR_CLAIM,
                source_reference=candidate.claim_id,
                summary=(
                    f"Comparative context only: earlier {candidate.claim_type[:100]} claim dated "
                    f"{candidate.incident_date}, collision {candidate.collision_type[:100]}, "
                    f"repair {candidate.repair_amount} {candidate.repair_currency[:10]}. "
                    "Similarity is not evidence of misconduct or a basis for copying a prior decision."
                ),
                structured_data=data,
            )
        )
        if len(results) == arguments.limit:
            break
    return evidence, results
