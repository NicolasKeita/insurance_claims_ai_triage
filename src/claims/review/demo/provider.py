"""Read prepared review snapshots and their explicitly allowed image assets."""

import json
import os
from pathlib import Path

from claims.review.models import ClaimReview, DemoClaim
from persistence.repositories import ClaimNotFoundError

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DEMO_DIR = ROOT / "artifacts/review_demo"
DEMO_FORMAT = "claim_review_demo_v1"


class DemoReviewService:
    """A read-only filesystem adapter: no extraction, inference or network on GET."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.environ.get("CLAIMS_REVIEW_DEMO_DIR", DEFAULT_DEMO_DIR)).resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Demo snapshots are unavailable. Run scripts/prepare_review_demo.py first.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != DEMO_FORMAT:
            raise ValueError("Unsupported review demo snapshot format")
        self.entries = {item["claim_id"]: item for item in manifest["claims"]}

    def list_demo_claims(self) -> list[DemoClaim]:
        return [
            DemoClaim(claim_id=item["claim_id"], title=item["title"], description=item["description"])
            for item in self.entries.values()
        ]

    def get_review(self, claim_id: str) -> ClaimReview:
        entry = self._entry(claim_id)
        snapshot = self._contained(entry["snapshot"])
        review = ClaimReview.model_validate_json(snapshot.read_text(encoding="utf-8"))
        if review.claim.claim_id != claim_id:
            raise ValueError("Prepared snapshot identity differs from its manifest entry")
        return review

    def image_path(self, claim_id: str, filename: str) -> Path | None:
        entry = self._entry(claim_id)
        relative = entry.get("images", {}).get(filename)
        if relative is None:
            return None
        try:
            path = self._contained(relative)
        except ValueError:
            return None
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} or not path.is_file():
            return None
        return path

    def _entry(self, claim_id):
        try:
            return self.entries[claim_id]
        except KeyError as error:
            raise ClaimNotFoundError(f"Claim {claim_id} does not exist in the prepared demo") from error

    def _contained(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Demo asset path leaves its configured snapshot directory")
        return path
