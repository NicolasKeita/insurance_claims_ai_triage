"""Explicit CPU-only preparation of the canonical controlled portfolio cases."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from claims.consistency import compare_claim_documents
from claims.enums import DocumentType
from claims.review.assembly import build_review
from claims.review.demo.analysis import document_review, partial_investigation
from claims.review.demo.provider import DEFAULT_DEMO_DIR, DEMO_FORMAT
from claims.review.models import ReviewSection
from evaluation.deterministic.documents import extract_case_documents
from evaluation.deterministic.fixtures import materialize_scenario_case
from evaluation.scenarios import DEFAULT_SCENARIOS_PATH, load_scenarios

MINIMAL_SCENARIOS = ("simple_coherent_claim", "additional_quote_damage", "missing_documents")
TITLES = {
    "simple_coherent_claim": "Coherent documents",
    "additional_quote_damage": "Additional quote damage",
    "missing_documents": "Missing documents",
}


def prepare_demo(output_dir: str | Path = DEFAULT_DEMO_DIR, observations: list[Path] | None = None):
    """Create source fixtures and document/rule outputs in a fresh directory.

    No evaluation runner, model, agent or service is invoked. Previously saved
    real observations can be imported explicitly afterwards, without inference.
    """
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Demo output directory is not empty; choose a new --output-dir to preserve snapshots.")
    root.mkdir(parents=True, exist_ok=True)
    entries = [_prepare_scenario(root, scenario) for scenario in load_scenarios(scenario_ids=MINIMAL_SCENARIOS)]
    if observations:
        from claims.review.demo.observations import import_observations

        imported = import_observations(root, observations)
        replacements = {item["claim_id"]: item for item in imported}
        entries = [replacements.pop(item["claim_id"], item) for item in entries] + list(replacements.values())
    manifest = {
        "format": DEMO_FORMAT,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": "data/evaluation/e2e_v2/scenarios.json",
        "source_sha256": hashlib.sha256(DEFAULT_SCENARIOS_PATH.read_bytes()).hexdigest(),
        "claims": entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _prepare_scenario(root, scenario):
    case = materialize_scenario_case(scenario, root / "fixtures" / scenario.scenario_id)
    extractions, ingestions = extract_case_documents(case)
    consistency = _consistency(extractions)
    sections = {
        "documents": document_review(case, extractions, ingestions),
        "investigation": partial_investigation(case, consistency),
    }
    sections["consistency"] = (
        ReviewSection(
            status="AVAILABLE", data=consistency.model_dump(mode="json"), source=f"consistency:{case.claim.claim_id}"
        )
        if consistency is not None
        else ReviewSection(
            status="UNAVAILABLE", reason="A comparison document is missing; matching fields cannot be established."
        )
    )
    review = build_review(case.claim, sections, mode="demo")
    snapshot = f"{scenario.scenario_id}.json"
    (root / snapshot).write_text(review.model_dump_json(indent=2), encoding="utf-8")
    return {
        "claim_id": case.claim.claim_id,
        "scenario_id": scenario.scenario_id,
        "title": TITLES[scenario.scenario_id],
        "description": "Controlled synthetic documents; CPU document analysis and partial deterministic signals.",
        "snapshot": snapshot,
        "images": {},
    }


def _consistency(extractions):
    kinds = (DocumentType.CLAIM_FORM, DocumentType.ACCIDENT_REPORT, DocumentType.REPAIR_QUOTE)
    if not all(kind in extractions for kind in kinds):
        return None
    return compare_claim_documents(*(extractions[kind] for kind in kinds))
