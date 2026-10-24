"""Validate declared visual references without loading detectors or running inference."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def visual_assets(scenario):
    """Return readable, fingerprinted images with explicit origin and reviewed part labels.

    Generated images are valid controlled inputs for real detector execution; their
    provenance must never describe them as photographs or real-world accuracy data.
    Labels are reference metadata, never detector inputs or replacement predictions.
    """
    from claims.enums import DamageType
    from vision.images import load_validated_image

    if not scenario.external_assets:
        raise ValueError("No annotated visual reference images configured for this scenario")
    validated = []
    for asset in scenario.external_assets:
        if not asset.get("path"):
            raise ValueError("Visual reference image needs a path")
        if asset.get("origin") not in {"ai_generated", "photograph"}:
            raise ValueError("Visual reference image origin must be ai_generated or photograph")
        if not isinstance(asset.get("provenance"), str) or not asset["provenance"].strip():
            raise ValueError("Visual reference image needs explicit provenance")
        annotation = asset.get("annotation")
        parts = annotation.get("damaged_parts") if isinstance(annotation, dict) else None
        if not isinstance(parts, list):
            raise ValueError("Visual reference image needs annotated damaged_parts")
        for part in parts:
            DamageType(part)
        path = (ROOT / asset["path"]).resolve()
        with load_validated_image(path):
            pass
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if asset.get("sha256") != digest:
            raise ValueError(f"Visual reference image SHA-256 mismatch: {path}")
        validated.append({**asset, "path": str(path), "sha256": digest})
    return validated
