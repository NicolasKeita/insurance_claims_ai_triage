"""Read prepared review snapshots and their explicitly allowed image assets."""

import json
import os
from pathlib import Path

from claims.review.models import ClaimReview, DemoClaim
from evaluation.deterministic.document_generation.identifiers import display_identifiers
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
        self.archive = None
        if manifest.get("format") in {"claim_analysis_archive_v1", "claim_analysis_collection_v1"}:
            from claims.review.archive.provider import ArchiveReviewService

            self.archive = ArchiveReviewService(self.root)
            self.entries = self.archive.entries
            return
        if manifest.get("format") != DEMO_FORMAT:
            raise ValueError("Unsupported review demo snapshot format")
        self.entries = {item["claim_id"]: item for item in manifest["claims"]}

    def list_demo_claims(self) -> list[DemoClaim]:
        if self.archive is not None:
            return self.archive.list_demo_claims()
        return [
            DemoClaim(claim_id=item["claim_id"], title=item["title"], description=item["description"])
            for item in self.entries.values()
        ]

    def get_review(self, claim_id: str, analysis_id=None) -> ClaimReview:
        if self.archive is not None:
            return display_identifiers(self.archive.get_review(claim_id, analysis_id))
        if analysis_id is not None:
            raise ClaimNotFoundError("A specific analysis requires an exported analysis archive")
        entry = self._entry(claim_id)
        snapshot = self._contained(entry["snapshot"])
        review = ClaimReview.model_validate_json(snapshot.read_text(encoding="utf-8"))
        if review.claim.claim_id != claim_id:
            raise ValueError("Prepared snapshot identity differs from its manifest entry")
        return display_identifiers(review)

    def image_path(self, claim_id: str, filename: str, analysis_id=None) -> Path | None:
        if self.archive is not None:
            return self.archive.image_path(claim_id, filename, analysis_id)
        if analysis_id is not None:
            raise ClaimNotFoundError("A specific analysis requires an exported analysis archive")
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

    def document_path(self, claim_id, filename, analysis_id=None):
        if self.archive is not None:
            return self.archive.document_path(claim_id, filename, analysis_id)
        if analysis_id is not None:
            raise ClaimNotFoundError("A specific analysis requires an exported analysis archive")
        self._entry(claim_id)
        return None

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
