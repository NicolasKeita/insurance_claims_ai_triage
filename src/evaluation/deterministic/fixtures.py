"""Generate controlled synthetic claim fixtures in an isolated directory."""

import json
import shutil
from decimal import Decimal
from pathlib import Path
from tempfile import mkdtemp

import pymupdf

from claims.case import ClaimCase
from claims.loader import load_claim_case
from evaluation.deterministic.documents import document_payloads, document_texts

ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC_PROVENANCE = "Generated controlled synthetic E2E fixture; no third-party imagery or real personal data."


def _pdf(path: Path, text: str):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((45, 45), text, fontsize=10)
    document.save(path)
    document.close()


def _scenario_payload(scenario, inputs):
    """Load the source claim and apply controlled fixture inputs to it."""
    source = Path(scenario.claim_source)
    source = source if source.is_absolute() else ROOT / source
    if source.is_dir():
        source /= "claim.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["claim_id"] = inputs.get("claim_id", f"DEMO-E2E-{scenario.scenario_id.upper()}")
    payload["customer"]["customer_id"] = f"DEMO-E2E-CUSTOMER-{scenario.scenario_id}"
    payload["policy"]["policy_id"] = f"DEMO-E2E-POLICY-{scenario.scenario_id}"
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


def _write_image(root, payload):
    from PIL import Image, ImageDraw

    images = root / "images"
    images.mkdir(exist_ok=True)
    with Image.new("RGB", (128, 128), "#d8dee3") as image:
        drawing = ImageDraw.Draw(image)
        drawing.rectangle((12, 12, 116, 116), outline="#304050", width=2)
        drawing.line((44, 44, 64, 62), fill="#8b3344", width=3)
        image.save(images / payload["images"][0])


def _copy_visual_assets(root, scenario, payload):
    from evaluation.visual_assets import visual_assets

    assets = visual_assets(scenario)
    if not assets:
        raise RuntimeError("Visual scenario requires validated image assets; no generated geometry fallback")
    images = root / "images"
    images.mkdir(exist_ok=True)
    copied = []
    for index, asset in enumerate(assets, start=1):
        filename = f"{index:02d}-{Path(asset['path']).name}"
        shutil.copyfile(asset["path"], images / filename)
        copied.append({**asset, "fixture_filename": filename})
    payload["images"] = [asset["fixture_filename"] for asset in copied]
    return copied


def _write_provenance(root, scenario, source, visual_assets):
    provenance = (
        "Controlled synthetic claim and documents; source image origin is recorded per asset."
        if visual_assets else SYNTHETIC_PROVENANCE
    )
    (root / "fixture_provenance.json").write_text(
        json.dumps(
            {
                "scenario_id": scenario.scenario_id,
                "source_claim": str(source),
                "provenance": provenance,
                "vision_semantics": (
                    "Validated source images copied unchanged for model inference; "
                    "image origin and annotation scope are recorded per asset."
                    if visual_assets else
                    "Generated pixels exercise geometry only; fake detections are explicit test inputs."
                ),
                "visual_assets": visual_assets,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


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
    visual_assets = _copy_visual_assets(root, scenario, payload) if use_visual_assets and scenario.vision_required else []
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
