"""Generate controlled synthetic claim fixtures in an isolated directory."""

import json
from decimal import Decimal
from pathlib import Path
from tempfile import mkdtemp

import pymupdf

from claims.case import ClaimCase
from claims.loader import load_claim_case
from evaluation.deterministic.document_generation import render_page
from evaluation.deterministic.document_generation.assets import (
    SYNTHETIC_PROVENANCE,
    _copy_visual_assets,
    _write_image,
    _write_provenance,
)
from evaluation.deterministic.document_generation.identifiers import fake_identifiers
from evaluation.deterministic.documents import document_payloads, document_texts

__all__ = ["ROOT", "SYNTHETIC_PROVENANCE", "materialize_scenario_case"]

ROOT = Path(__file__).resolve().parents[3]


def _pdf(path: Path, text: str):
    with pymupdf.open() as document:
        render_page(document, text, kind=path.stem)
        document.save(path, no_new_id=True, garbage=4, deflate=True)


def _scenario_payload(scenario, inputs):
    """Load the source claim and apply controlled fixture inputs to it."""
    source = Path(scenario.claim_source)
    source = source if source.is_absolute() else ROOT / source
    if source.is_dir():
        source /= "claim.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["claim_id"] = inputs.get("claim_id", f"DEMO-E2E-{scenario.scenario_id.upper()}")
    customer_id, policy_id = fake_identifiers(scenario.scenario_id)
    payload["customer"]["customer_id"] = customer_id
    payload["policy"]["policy_id"] = policy_id
    payload["declared_damage"] = inputs.get("declared_damage", ["FRONT_BUMPER"])
    quoted = inputs.get("quote_damage", payload["declared_damage"])
    payload["images"] = ["synthetic_geometry.png"] if scenario.vision_required else []
    return source, payload, quoted


def _apply_overrides(payload, inputs):
    for key, value in inputs.get("claim_overrides", {}).items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key].update(value)
        else:
            payload[key] = value


def materialize_scenario_case(scenario, root_dir=None, *, use_visual_assets=False) -> ClaimCase:
    """Create demo PDFs and isolated image fixtures, optionally copying source assets.

    The caller owns cleanup of ``root_dir``. If omitted, a fresh temporary dir is
    returned through ``ClaimCase.root_dir``. No inference or fake model is used.
    Real callers opt into validated image assets; deterministic callers retain
    the generated geometry fixture independently of external source files.

    General overrides are fixture inputs, never inferred from expectations.

    Amount arithmetic remains exact and the real quote parser verifies it.
    """
    root = Path(root_dir) if root_dir is not None else Path(mkdtemp(prefix="claims-e2e-"))
    root.mkdir(parents=True, exist_ok=True)
    inputs = scenario.inputs
    source, payload, quoted = _scenario_payload(scenario, inputs)
    filenames, missing, entries = document_payloads(inputs)
    payload["documents"] = entries
    amount = Decimal(str(inputs.get("repair_amount", 1500)))
    payload["repair_estimate"]["amount"] = str(amount)
    _apply_overrides(payload, inputs)
    visual_assets = (
        _copy_visual_assets(root, scenario, payload) if use_visual_assets and scenario.vision_required else []
    )
    texts = document_texts(payload, inputs, quoted, amount)
    docs = root / "documents"
    docs.mkdir(exist_ok=True)
    for kind, filename in filenames.items():
        if kind not in missing:
            _pdf(docs / filename, texts[kind])
    if payload["images"] and not visual_assets:
        _write_image(root, payload)
    (root / "claim.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_provenance(root, scenario, source, visual_assets)
    return load_claim_case(root)
