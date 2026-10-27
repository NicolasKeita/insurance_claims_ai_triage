"""Original claim photographs are served only from an explicit trusted root."""

import os
from pathlib import Path
from urllib.parse import quote


class ClaimImageStore:
    def __init__(self, asset_root=None):
        configured = asset_root or os.environ.get("CLAIMS_REVIEW_ASSET_ROOT")
        self.root = Path(configured).resolve() if configured else None

    def path(self, claim, filename):
        if self.root is None or not filename or filename not in claim.images:
            return None
        if not _plain_filename(claim.claim_id) or not _plain_filename(filename):
            return None
        candidate = (self.root / claim.claim_id / "images" / filename).resolve()
        claim_root = (self.root / claim.claim_id).resolve()
        if (
            not claim_root.is_relative_to(self.root)
            or not candidate.is_relative_to(claim_root)
            or not candidate.is_file()
        ):
            return None
        return candidate if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else None

    def annotate(self, claim, images):
        for image in images:
            filename = image.get("filename") or image.get("damage_assessment", {}).get("filename")
            available = self.path(claim, filename) is not None
            image["image_url"] = (
                f"/v1/claims/{quote(claim.claim_id, safe='')}/review/images/{quote(filename, safe='')}"
                if available
                else None
            )
            image["image_unavailable_reason"] = (
                None if available else "The original photograph is not available under the configured claim asset root."
            )


def _plain_filename(value):
    return value not in {".", ".."} and not any(char in value for char in "/\\:")
