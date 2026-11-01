"""Synthetic fixture image materialization and recorded source provenance."""

import json
import shutil
from pathlib import Path

SYNTHETIC_PROVENANCE = "Generated controlled synthetic E2E fixture; no third-party imagery or real personal data."


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
        if visual_assets
        else SYNTHETIC_PROVENANCE
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
                    if visual_assets
                    else "Generated pixels exercise geometry only; fake detections are explicit test inputs."
                ),
                "visual_assets": visual_assets,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
