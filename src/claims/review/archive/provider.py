"""Read an individual portable analysis or an exported analysis collection."""

import hashlib
from pathlib import Path
from urllib.parse import quote

from claims.review.archive.assets import IMAGE_SUFFIXES, plain_name
from claims.review.archive.storage.catalogue import ARCHIVE_FORMAT, COLLECTION_FORMAT
from claims.review.archive.storage.validation import contained, read_json, verified_asset
from claims.review.models import ClaimReview, DemoClaim
from persistence.repositories.errors import ClaimNotFoundError


class ArchiveReviewService:
    def __init__(self, root):
        self.root = Path(root).absolute()
        manifest = read_json(contained(self.root, "manifest.json"))
        if manifest.get("format") == ARCHIVE_FORMAT:
            items = [{"manifest": "manifest.json", **manifest}]
        elif manifest.get("format") == COLLECTION_FORMAT:
            items = manifest["analyses"]
        else:
            raise ValueError("Unsupported analysis archive format")
        self.analyses = {}
        self.entries = {}
        for item in items:
            bundle = contained(self.root, item["manifest"])
            data = read_json(bundle)
            if data.get("format") != ARCHIVE_FORMAT:
                raise ValueError("Unsupported analysis archive format")
            for key in ("claim_id", "analysis_id"):
                plain_name(data[key], key)
                if item.get(key) != data[key]:
                    raise ValueError("Analysis archive identity differs from the collection entry")
            entry = {**data, "manifest_path": bundle}
            key = (data["claim_id"], data["analysis_id"])
            if key in self.analyses:
                raise ValueError("Duplicate analysis identity in collection")
            self.analyses[key] = entry
            self.entries[data["claim_id"]] = entry

    def list_demo_claims(self):
        return [
            DemoClaim(
                claim_id=item["claim_id"],
                title=item["title"],
                description=f"Archived analysis {item['analysis_id']}; results and copied source files.",
            )
            for item in self.entries.values()
        ]

    def _entry(self, claim_id, analysis_id=None):
        item = self.analyses.get((claim_id, analysis_id)) if analysis_id else self.entries.get(claim_id)
        if item is None:
            raise ClaimNotFoundError(f"Claim {claim_id} has no matching exported analysis")
        return item

    def get_review(self, claim_id, analysis_id=None):
        entry = self._entry(claim_id, analysis_id)
        root = entry["manifest_path"].parent
        path = contained(root, entry["review_file"])
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry.get("review_sha256"):
            raise ValueError("Archived review checksum does not match its manifest")
        review = ClaimReview.model_validate_json(payload)
        if review.claim.claim_id != claim_id:
            raise ValueError("Archived review identity differs from its manifest")
        for image in (review.vision.data or {}).get("images", []):
            filename = image.get("filename")
            image["image_url"] = self._url(entry, filename, "image")
            image["image_unavailable_reason"] = (
                None if image["image_url"] else ("The original photograph is not included in this archive.")
            )
        for document in (review.documents.data or {}).get("items", []):
            filename = document.get("filename")
            document["document_url"] = self._url(entry, filename, "document")
            document["document_unavailable_reason"] = (
                None if document["document_url"] else ("The original PDF is not included in this archive.")
            )
        return review

    def _url(self, entry, filename, kind):
        if self._asset(entry, filename, kind) is None:
            return None
        segment = "images" if kind == "image" else "documents"
        return (
            f"/v1/claims/{quote(entry['claim_id'], safe='')}/review/{segment}/{quote(filename, safe='')}"
            f"?analysis={quote(entry['analysis_id'], safe='')}"
        )

    @staticmethod
    def _asset(entry, filename, kind):
        suffixes = IMAGE_SUFFIXES if kind == "image" else {".pdf"}
        if not filename or Path(filename).suffix.lower() not in suffixes:
            return None
        item = next(
            (
                item
                for item in entry["files"]
                if item["kind"] == kind and item["filename"] == filename and item.get("status") == "COPIED"
            ),
            {},
        )
        return verified_asset(entry["manifest_path"].parent, item)

    def image_path(self, claim_id, filename, analysis_id=None):
        return self._asset(self._entry(claim_id, analysis_id), filename, "image")

    def document_path(self, claim_id, filename, analysis_id=None):
        return self._asset(self._entry(claim_id, analysis_id), filename, "document")
