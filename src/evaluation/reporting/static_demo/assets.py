"""Copy verified demo sources and localize recorded review asset references."""

import hashlib
import shutil
from urllib.parse import quote

from claims.review.archive.assets import plain_name
from evaluation.reporting.static_demo.documents import restyle_document


def static_review(service, review, entry, root):
    """Return JSON for the existing review, with standalone static asset URLs."""
    bundle = f"{plain_name(review.claim.claim_id)}_{plain_name(entry['analysis_id'])}"
    urls = {}
    provenance = []
    for item in entry["files"]:
        record = {
            "kind": item["kind"],
            "filename": item["filename"],
            "status": item["status"],
            "original_sha256": item.get("sha256"),
            "presentation_sha256": None,
            "path": None,
        }
        provenance.append(record)
        if item["status"] != "COPIED":
            continue
        kind, filename = item["kind"], plain_name(item["filename"])
        if kind not in {"document", "image"}:
            raise ValueError("Unsupported static demo source asset kind")
        resolver = service.document_path if kind == "document" else service.image_path
        source = resolver(review.claim.claim_id, filename, entry["analysis_id"])
        if source is None:
            raise ValueError(f"Demo source checksum is invalid or the file is missing: {filename}")
        folder = "documents" if kind == "document" else "images"
        target = root / folder / bundle / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if kind == "document":
            restyle_document(source, target, item["document_type"])
        else:
            shutil.copyfile(source, target)
        record.update(
            presentation_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            path=target.relative_to(root).as_posix(),
        )
        urls[kind, filename] = f"demo/{folder}/{bundle}/{quote(filename, safe='')}"
    return _localize(review.model_dump(mode="json"), urls), provenance


def _localize(value, urls):
    if isinstance(value, list):
        return [_localize(item, urls) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _localize(item, urls) for key, item in value.items()}
    for kind in ("image", "document"):
        key = f"{kind}_url"
        if key in result:
            result[key] = urls.get((kind, result.get("filename")))
    return result
