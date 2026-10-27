"""Stable links into the evidence actually present in a review snapshot."""

import re

from claims.review.models import ReviewEvidence

SOURCE_SECTIONS = {
    "CLAIM_DATA": "overview",
    "HISTORY": "history",
    "TRIAGE_ASSESSMENT": "triage",
    "ANOMALY_ASSESSMENT": "anomaly",
    "INVESTIGATION_ASSESSMENT": "investigation",
    "VISUAL_ASSESSMENT": "vision",
    "SIMILAR_CLAIM": "similar_claims",
    "POLICY_CHUNK": "knowledge",
    "PROCEDURE_CHUNK": "knowledge",
}


def build_evidence(claim, sections):
    registry = {}

    def add(identifier, section, title, data=None, source=None):
        registry.setdefault(
            identifier, ReviewEvidence(id=identifier, section=section, title=title, data=data, source=source)
        )

    add(f"claim:{claim.claim_id}", "overview", f"Claim {claim.claim_id}", source=claim.claim_id)
    for name, section in sections.items():
        if section.data is not None and section.source:
            add(section.source, name, f"Stored {name.replace('_', ' ')}", section.data, section.source)
    documents = _data(sections["documents"]).get("items", [])
    for item in documents:
        add(f"document:{item['type']}", "documents", item.get("filename") or item["type"], item, item.get("source"))
    for signal in _data(sections["investigation"]).get("signals", []):
        add(f"signal:{signal['code']}", "investigation", signal.get("title", signal["code"]), signal)
    for image in _data(sections["vision"]).get("images", []):
        filename = image.get("filename") or image.get("damage_assessment", {}).get("filename", "image")
        for index, observation in enumerate(image.get("evidence", []), 1):
            add(f"vision:{filename}:obs_{index}", "vision", f"{filename} observation {index}", observation)
    for source in _data(sections["knowledge"]).get("sources", []):
        add(f"knowledge:{source['chunk_id']}", "knowledge", source.get("title", source["chunk_id"]), source)
    for similar in _data(sections["similar_claims"]).get("results", []):
        add(f"similar:{similar['claim_id']}", "similar_claims", f"Similar claim {similar['claim_id']}", similar)
    agent = _data(sections["agent"])
    for item in agent.get("evidence", []):
        add(
            item["evidence_id"],
            SOURCE_SECTIONS.get(item["source_type"], "agent"),
            evidence_title(item),
            item.get("structured_data"),
            item.get("source_reference"),
        )
    if agent.get("run_id"):
        add(f"agent:{agent['run_id']}", "agent", "Stored agent recommendation", agent.get("recommendation"))
    return list(registry.values())


def _data(section):
    return section.data if isinstance(section.data, dict) else {}


def evidence_title(item):
    """Source labels remain readable even when a historical summary embeds JSON."""
    return item.get("source_type", "EVIDENCE").replace("_", " ").capitalize()


def safe_summary(item):
    summary = item.get("summary", "")
    if "{" in summary or summary.lstrip().startswith("[") or re.search(r"(?:^|\s|[\"'(])[A-Za-z]:[\\/]", summary):
        return f"{evidence_title(item)}. See the structured assessment evidence."
    return summary
