"""Resolve explicitly selected original assets for a stored review."""

import json
from pathlib import Path

from claims.models import Claim
from claims.review.archive.assets import plain_name


def case_sources(review, case_dir):
    plain_name(review.claim.claim_id, "claim identifier")
    root = Path(case_dir).absolute()
    if any(part.is_symlink() for part in (root, *root.parents)):
        raise ValueError("Source case directory symlinks are not allowed")
    claim_file = root / "claim.json"
    if claim_file.is_symlink():
        raise ValueError("Source claim file symlinks are not allowed")
    if claim_file.exists():
        source = Claim.model_validate_json(claim_file.read_text(encoding="utf-8"))
        if source != review.claim:
            raise ValueError("Source case facts differ from the stored review; select the matching original case")
    documents = {
        plain_name(document.filename): root / "documents" / document.filename
        for document in review.claim.documents
        if document.filename and document.available
    }
    images = {plain_name(filename): root / "images" / filename for filename in review.claim.images}
    return documents, images


def selected_cases(paths):
    result = {}
    for path in paths:
        root = Path(path).absolute()
        source = root / "claim.json"
        if source.is_symlink() or any(part.is_symlink() for part in (root, *root.parents)):
            raise ValueError("Source case symlinks are not allowed")
        claim = Claim.model_validate_json(source.read_text(encoding="utf-8"))
        if claim.claim_id in result:
            raise ValueError("Select only one original source directory for each claim")
        result[claim.claim_id] = root
    return result


def prepared_sources(service, review, cases, asset_root=None):
    entry = service.entries[review.claim.claim_id]
    case = cases.get(review.claim.claim_id)
    if case is None and asset_root:
        case = Path(asset_root) / plain_name(review.claim.claim_id, "claim identifier")
    if case is None and entry.get("scenario_id"):
        case = service._contained(f"fixtures/{plain_name(entry['scenario_id'])}")
    documents, images = case_sources(review, case) if case is not None else ({}, {})
    for filename in review.claim.images:
        registered = service.image_path(review.claim.claim_id, filename)
        if registered:
            images[filename] = registered
    if getattr(service, "archive", None) is not None:
        for document in review.claim.documents:
            if document.filename and (path := service.document_path(review.claim.claim_id, document.filename)):
                documents[document.filename] = path
    return documents, images


def read_review(path):
    from claims.review.models import ClaimReview

    source = Path(path)
    return ClaimReview.model_validate(json.loads(source.read_text(encoding="utf-8")))
