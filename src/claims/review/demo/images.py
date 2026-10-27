"""Allow source imagery only when canonical metadata and recorded hashes agree."""

import hashlib
import shutil
from pathlib import Path
from urllib.parse import quote

from evaluation.visual_assets import visual_assets


def import_images(root, scenario, observation, payload):
    assets = visual_assets(scenario) if scenario.vision_required else []
    recorded = observation.get("provenance", {}).get("visual_assets", [])
    allowed, images = {}, {}
    for index, asset in enumerate(assets, 1):
        filename = f"{index:02d}-{Path(asset['path']).name}"
        record = next((item for item in recorded if item.get("fixture_filename") == filename), None)
        if record and record.get("sha256") == asset["sha256"]:
            allowed[filename] = asset
    for image in payload["images"]:
        filename = image["filename"]
        asset = allowed.get(filename)
        if asset is None:
            image["image_url"] = None
            image["image_unavailable_reason"] = (
                "The original image could not be matched to an authorized canonical demo asset."
            )
            continue
        _copy_image(root, scenario, image, asset)
        images[filename] = f"images/{scenario.scenario_id}/{filename}"
        claim_id, encoded_filename = quote(payload["claim_id"], safe=""), quote(filename, safe="")
        image["image_url"] = f"/v1/claims/{claim_id}/review/images/{encoded_filename}"
        image["image_unavailable_reason"] = None
        image["provenance"] = {key: asset.get(key) for key in ("origin", "provenance", "sha256")}
    return images


def _copy_image(root, scenario, image, asset):
    source = Path(asset["path"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != asset["sha256"]:
        raise ValueError("Demo source image SHA256 differs from recorded visual provenance")
    target = root / "images" / scenario.scenario_id / image["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
