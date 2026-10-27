"""Common presentation mapping for persisted and controlled demo snapshots."""

from pathlib import PurePosixPath, PureWindowsPath

from claims.review.evidence import build_evidence, safe_summary
from claims.review.models import ClaimReview, ReviewAudit, ReviewSection

SECTION_NAMES = (
    "documents",
    "consistency",
    "triage",
    "anomaly",
    "history",
    "investigation",
    "vision",
    "visual_consistency",
    "similar_claims",
    "knowledge",
    "agent",
)


def safe_data(value):
    """Omit execution internals and local asset paths from public review data."""
    if isinstance(value, dict):
        public = {
            key: safe_data(item)
            for key, item in value.items()
            if key
            not in {
                "trace",
                "planner_reason",
                "tool_args",
                "asset_reference",
                "file_path",
                "checkpoint_path",
                "signal_details",
            }
        }
        if value.get("evidence_id") and value.get("source_type"):
            public["summary"] = safe_summary(value)
        return public
    if isinstance(value, list):
        return [safe_data(item) for item in value]
    if isinstance(value, str) and (PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()):
        return value if value.startswith("/v1/") else PureWindowsPath(value).name
    return value


def build_review(claim, sections=None, *, mode="persisted") -> ClaimReview:
    """Assemble already computed facts; never invoke a business calculation."""
    defaults = {
        name: ReviewSection(status="NOT_EXECUTED", reason=f"No stored {name.replace('_', ' ')} result is available.")
        for name in SECTION_NAMES
    }
    defaults["documents"] = ReviewSection(
        status="AVAILABLE" if claim.documents else "NO_EVIDENCE",
        reason=None if claim.documents else "No documents are recorded for this claim.",
        source=f"claim:{claim.claim_id}",
        data={
            "items": [
                {
                    **document.model_dump(mode="json"),
                    "extraction_method": None,
                    "source": f"claim:{claim.claim_id}",
                    "extracted_data": None,
                }
                for document in claim.documents
            ],
            "extractions": {},
        },
    )
    defaults["consistency"] = ReviewSection(
        status="UNAVAILABLE", reason="Document extraction and consistency snapshots are not stored with this claim."
    )
    defaults["history"] = ReviewSection(status="UNAVAILABLE", reason="Historical context is unavailable.")
    defaults.update(sections or {})
    prepared = {
        name: section.model_copy(update={"data": safe_data(section.data)}) for name, section in defaults.items()
    }
    evidence = build_evidence(claim, prepared)
    return ClaimReview(claim=claim, **prepared, evidence=evidence, audit=_audit(claim, prepared, evidence), mode=mode)


def _audit(claim, sections, evidence):
    models, policies, timestamps = {}, {}, {}
    for name, section in sections.items():
        if section.updated_at is not None:
            timestamps[name] = section.updated_at.isoformat()
        data = section.data if isinstance(section.data, dict) else {}
        if name in {"triage", "anomaly"} and data:
            models[name] = data.get("model_version")
            models[f"{name}_model"] = data.get("registered_model")
        if name == "investigation" and data:
            policies["investigation"] = data.get("policy_version")
        if name == "vision" and data:
            for kind in ("damage", "part"):
                metadata = data.get(f"{kind}_model_metadata", {})
                models[f"{kind}_model"] = metadata.get("identifier")
                models[kind] = metadata.get("version")
            policies["vision_fusion"] = data.get("fusion_policy_version") or data.get("fusion_policy", {}).get(
                "version"
            )
        if name == "agent" and data:
            models["agent"] = data.get("agent_version")
            policies["prompt"] = data.get("prompt_version")
            for key in ("created_at", "completed_at"):
                if data.get(key):
                    timestamps[f"agent_{key}"] = data[key]
    return ReviewAudit(
        timestamps=timestamps,
        model_versions=models,
        policy_versions=policies,
        evidence_ids=[item.id for item in evidence],
        source_documents=[document.filename for document in claim.documents if document.filename],
        execution_statuses={name: section.status for name, section in sections.items()},
    )
