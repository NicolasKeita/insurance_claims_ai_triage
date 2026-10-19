"""Provenance checks over observed documents, quotes and model versions."""

from evaluation.assertions.matching import fact_at
from evaluation.assertions.recording import Check


def _document_sources(observation):
    sources = observation.provenance.get("documents", [])
    return list(sources.values()) if isinstance(sources, dict) else sources


def check_document_provenance(recorder):
    """The source snapshot, not an explanatory string, anchors document findings."""
    if recorder.skipped("document"):
        return
    sources = _document_sources(recorder.observation)
    valid = bool(sources) and all(
        isinstance(source, dict)
        and source.get("filename")
        and source.get("document_type")
        and isinstance(source.get("page"), int)
        and source["page"] >= 1
        for source in sources
    )
    recorder.record(
        Check("provenance:documents", "document", "source filename, document type and page", sources, valid)
    )


def check_quote_provenance(recorder):
    """A quote signal needs its quote source and exact damage evidence."""
    observation = recorder.observation
    if "ADDITIONAL_QUOTE_DAMAGE" not in observation.signals:
        return
    quote = [source for source in _document_sources(observation) if "QUOTE" in str(source.get("document_type", ""))]
    evidence = [
        item.get("evidence", {})
        for item in (observation.investigation or {}).get("signals", [])
        if item.get("code") == "ADDITIONAL_QUOTE_DAMAGE"
    ]
    actual_damage = fact_at(observation.facts, "consistency.additional_quote_damage", [])
    supported = {damage for item in evidence for damage in item.get("damage_types", [])}
    recorder.record(
        Check(
            "provenance:quote_signal",
            "investigation",
            "quote source and exact additional damage evidence",
            {"sources": quote, "evidence": evidence},
            bool(quote) and bool(evidence) and set(actual_damage) == supported,
        )
    )


def check_model_provenance(recorder):
    """Recorded model versions and image metadata must be present."""
    observation = recorder.observation
    if not recorder.skipped("anomaly"):
        version = observation.model_versions.get("anomaly")
        recorder.record(Check("provenance:anomaly_model", "anomaly", "recorded model/version", version, bool(version)))
    if observation.vision:
        images = observation.vision.get("images", [])
        metadata_ok = bool(images) and bool(observation.vision.get("damage_model_metadata", {}).get("version"))
        metadata_ok = metadata_ok and all(
            image.get("part_assessment", {}).get("model_metadata", {}).get("version")
            and image.get("damage_assessment", {}).get("asset_reference")
            for image in images
        )
        recorder.record(
            Check(
                "provenance:vision",
                "vision",
                "image + both model versions + validated geometry",
                bool(metadata_ok),
                bool(metadata_ok),
            )
        )


def check_provenance(recorder):
    """Run every provenance check for the recorded observation."""
    check_document_provenance(recorder)
    check_quote_provenance(recorder)
    check_model_provenance(recorder)
