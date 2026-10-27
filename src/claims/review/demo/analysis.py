"""Compose existing document extraction and independent investigation translators."""

from datetime import datetime, timezone
from types import SimpleNamespace

from claims.investigation.models import InvestigationAssessment
from claims.investigation.policy import DEFAULT_INVESTIGATION_POLICY
from claims.investigation.signals import signals_from_completeness, signals_from_consistency
from claims.review.models import ReviewSection
from ml.features import calculate_document_completeness


def document_review(case, extractions, ingestions):
    items = []
    for entry in case.claim.documents:
        extraction = extractions.get(entry.type)
        ingestion = ingestions.get(entry.filename)
        items.append(
            {
                **entry.model_dump(mode="json"),
                "extraction_method": ingestion.method.value if ingestion is not None else None,
                "source": "Controlled synthetic benchmark fixture",
                "extracted_data": extraction.model_dump(mode="json") if extraction is not None else None,
            }
        )
    return ReviewSection(
        status="AVAILABLE",
        data={
            "items": items,
            "extractions": {kind.value: value.model_dump(mode="json") for kind, value in extractions.items()},
        },
        source=f"documents:{case.claim.claim_id}",
        updated_at=datetime.now(timezone.utc),
    )


def partial_investigation(case, consistency):
    count, missing, ratio = calculate_document_completeness(case)
    completeness = SimpleNamespace(
        document_count=count, missing_document_count=missing, document_completeness_ratio=ratio
    )
    policy = DEFAULT_INVESTIGATION_POLICY
    signals = (
        *(signals_from_consistency(consistency) if consistency is not None else ()),
        *signals_from_completeness(completeness, policy),
    )
    score = policy.score_signals(signals)
    result = InvestigationAssessment(
        review_score=score,
        review_priority=policy.priority_for_score(score),
        review_recommended=policy.review_recommended_for_score(score),
        policy_version=policy.version,
        signals=signals,
    )
    return ReviewSection(
        status="AVAILABLE",
        data={
            **result.model_dump(mode="json"),
            "partial": True,
            "unavailable_inputs": ["anomaly", "history", "vision"],
        },
        reason=(
            "Partial prioritization from document and completeness evidence only; "
            "anomaly, history and vision not executed."
        ),
        source=f"investigation:{case.claim.claim_id}",
        updated_at=datetime.now(timezone.utc),
    )
